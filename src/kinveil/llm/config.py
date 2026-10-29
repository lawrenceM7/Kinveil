from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

_PASSWORD_CHECK_CONTEXT = b"kinveil-chat-master-password-v1"
_SCRYPT_LENGTH = 32
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1


@dataclass(frozen=True)
class ChatModelConfig:
    id: str
    name: str
    base_url: str
    model: str
    timeout_seconds: float = 120.0
    maximum_output_tokens: int = 1024
    temperature: float = 0.3

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.name.strip():
            raise ValueError("chat model id and name cannot be empty")
        if not self.base_url.strip().startswith(("https://", "http://")):
            raise ValueError("chat model URL must begin with http:// or https://")
        if not self.model.strip():
            raise ValueError("chat model name cannot be empty")
        if self.timeout_seconds <= 0 or self.maximum_output_tokens <= 0:
            raise ValueError("chat model limits must be positive")


class ChatConfigStore:
    """Stores non-secret model settings in the local user-data directory."""

    def __init__(self, database: Path) -> None:
        self.database = database
        database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_models (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    base_url TEXT NOT NULL,
                    model TEXT NOT NULL,
                    timeout_seconds REAL NOT NULL,
                    maximum_output_tokens INTEGER NOT NULL,
                    temperature REAL NOT NULL,
                    active INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_chat_model
                ON chat_models(active) WHERE active = 1;
                """
            )
        _restrict_file_permissions(database)

    def save(self, config: ChatModelConfig, *, active: bool = True) -> None:
        with self._connect() as connection:
            if active:
                connection.execute("UPDATE chat_models SET active = 0 WHERE active = 1")
            connection.execute(
                """
                INSERT INTO chat_models(
                    id, name, base_url, model, timeout_seconds,
                    maximum_output_tokens, temperature, active, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    base_url = excluded.base_url,
                    model = excluded.model,
                    timeout_seconds = excluded.timeout_seconds,
                    maximum_output_tokens = excluded.maximum_output_tokens,
                    temperature = excluded.temperature,
                    active = excluded.active,
                    updated_at = excluded.updated_at
                """,
                (
                    config.id,
                    config.name,
                    config.base_url.rstrip("/"),
                    config.model,
                    config.timeout_seconds,
                    config.maximum_output_tokens,
                    config.temperature,
                    int(active),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def active(self) -> ChatModelConfig | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, name, base_url, model, timeout_seconds,
                       maximum_output_tokens, temperature
                FROM chat_models WHERE active = 1
                """
            ).fetchone()
        return ChatModelConfig(*row) if row else None

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database)


class SecretStore(Protocol):
    def get(self, account: str) -> str | None: ...

    def set(self, account: str, secret: str) -> None: ...


class SecretStoreLockedError(RuntimeError):
    pass


class InvalidMasterPasswordError(ValueError):
    pass


class EncryptedSQLiteSecretStore:
    """AES-GCM secret storage unlocked by an in-memory Scrypt-derived key."""

    def __init__(self, database: Path) -> None:
        self.database = database
        database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_secret_settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    salt BLOB NOT NULL,
                    password_verifier BLOB NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS encrypted_chat_secrets (
                    account TEXT PRIMARY KEY,
                    nonce BLOB NOT NULL,
                    ciphertext BLOB NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
        _restrict_file_permissions(database)
        self._key: bytearray | None = None

    @property
    def initialized(self) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM chat_secret_settings WHERE id = 1"
            ).fetchone()
        return row is not None

    @property
    def unlocked(self) -> bool:
        return self._key is not None

    def unlock(self, password: str, *, create_if_missing: bool = False) -> None:
        if not password:
            raise InvalidMasterPasswordError("master password cannot be empty")
        if create_if_missing and len(password) < 8:
            raise InvalidMasterPasswordError(
                "master password must contain at least 8 characters"
            )
        with self._connect() as connection:
            row = connection.execute(
                "SELECT salt, password_verifier FROM chat_secret_settings WHERE id = 1"
            ).fetchone()
            if row is None:
                if not create_if_missing:
                    raise InvalidMasterPasswordError("master password has not been created")
                salt = os.urandom(16)
                key = _derive_key(password, salt)
                connection.execute(
                    """
                    INSERT INTO chat_secret_settings(
                        id, salt, password_verifier, created_at
                    ) VALUES (1, ?, ?, ?)
                    """,
                    (
                        salt,
                        _password_verifier(key),
                        datetime.now(UTC).isoformat(),
                    ),
                )
            else:
                salt, expected = bytes(row[0]), bytes(row[1])
                key = _derive_key(password, salt)
                if not hmac.compare_digest(_password_verifier(key), expected):
                    raise InvalidMasterPasswordError("incorrect master password")
        self.lock()
        self._key = bytearray(key)
        self._migrate_plaintext_secrets()

    def lock(self) -> None:
        if self._key is not None:
            for index in range(len(self._key)):
                self._key[index] = 0
        self._key = None

    def get(self, account: str) -> str | None:
        key = self._require_key()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT nonce, ciphertext FROM encrypted_chat_secrets
                WHERE account = ?
                """,
                (account,),
            ).fetchone()
        if row is None:
            return None
        try:
            plaintext = AESGCM(key).decrypt(
                bytes(row[0]), bytes(row[1]), account.encode("utf-8")
            )
        except InvalidTag as exc:
            raise RuntimeError("stored API key failed authentication") from exc
        return plaintext.decode("utf-8")

    def set(self, account: str, secret: str) -> None:
        key = self._require_key()
        clean = secret.strip()
        if not clean:
            raise ValueError("API key cannot be empty")
        nonce = os.urandom(12)
        ciphertext = AESGCM(key).encrypt(
            nonce, clean.encode("utf-8"), account.encode("utf-8")
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO encrypted_chat_secrets(
                    account, nonce, ciphertext, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(account) DO UPDATE SET
                    nonce = excluded.nonce,
                    ciphertext = excluded.ciphertext,
                    updated_at = excluded.updated_at
                """,
                (account, nonce, ciphertext, datetime.now(UTC).isoformat()),
            )

    def delete(self, account: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM encrypted_chat_secrets WHERE account = ?", (account,)
            )
        return cursor.rowcount > 0

    def _require_key(self) -> bytes:
        if self._key is None:
            raise SecretStoreLockedError("chat secrets are locked")
        return bytes(self._key)

    def _migrate_plaintext_secrets(self) -> None:
        with self._connect() as connection:
            legacy_exists = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'chat_secrets'
                """
            ).fetchone()
            rows = (
                connection.execute("SELECT account, api_key FROM chat_secrets").fetchall()
                if legacy_exists
                else ()
            )
        if not legacy_exists:
            return
        for account, api_key in rows:
            self.set(str(account), str(api_key))
        with self._connect() as connection:
            connection.execute("DROP TABLE chat_secrets")
        with self._connect() as connection:
            connection.execute("VACUUM")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database)

    def __del__(self) -> None:
        if getattr(self, "_key", None) is not None:
            self.lock()


class MemorySecretStore:
    """Test double that never writes secrets to disk."""

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def get(self, account: str) -> str | None:
        return self._values.get(account)

    def set(self, account: str, secret: str) -> None:
        self._values[account] = secret


def _restrict_file_permissions(path: Path) -> None:
    if os.name == "posix":
        path.chmod(0o600)


def _derive_key(password: str, salt: bytes) -> bytes:
    return Scrypt(
        salt=salt,
        length=_SCRYPT_LENGTH,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
    ).derive(password.encode("utf-8"))


def _password_verifier(key: bytes) -> bytes:
    return hmac.new(key, _PASSWORD_CHECK_CONTEXT, hashlib.sha256).digest()
