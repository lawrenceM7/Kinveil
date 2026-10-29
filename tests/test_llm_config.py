from __future__ import annotations

import json
import sqlite3

import pytest

from kinveil.llm import (
    ChatConfigStore,
    ChatMessage,
    ChatModelConfig,
    EncryptedSQLiteSecretStore,
    InvalidMasterPasswordError,
    LLMRequest,
    MemorySecretStore,
    OpenAICompatibleLanguageModel,
    SecretStoreLockedError,
    ensure_chat_setup,
)


def _config(**changes) -> ChatModelConfig:
    values = {
        "id": "primary",
        "name": "Primary",
        "base_url": "https://llm.example/v1",
        "model": "example-chat",
        "timeout_seconds": 10.0,
        "maximum_output_tokens": 300,
        "temperature": 0.2,
    }
    values.update(changes)
    return ChatModelConfig(**values)


def test_chat_config_and_key_are_encrypted_in_local_sqlite(tmp_path) -> None:
    database = tmp_path / "chat.sqlite3"
    configs = ChatConfigStore(database)
    secrets = EncryptedSQLiteSecretStore(database)
    secrets.unlock("correct horse battery staple", create_if_missing=True)
    configs.save(_config())
    secrets.set("primary", "super-secret-key")

    restored = ensure_chat_setup(configs, secrets, show_dialog=False)

    assert restored is not None
    assert restored.config.model == "example-chat"
    assert restored.api_key == "super-secret-key"
    contents = database.read_bytes()
    assert b"super-secret-key" not in contents
    assert b"correct horse battery staple" not in contents

    secrets.lock()
    with pytest.raises(SecretStoreLockedError):
        secrets.get("primary")
    secrets.unlock("correct horse battery staple")
    assert secrets.get("primary") == "super-secret-key"


def test_wrong_master_password_is_rejected(tmp_path) -> None:
    secrets = EncryptedSQLiteSecretStore(tmp_path / "chat.sqlite3")
    secrets.unlock("right password value", create_if_missing=True)
    secrets.set("primary", "secret")
    secrets.lock()

    with pytest.raises(InvalidMasterPasswordError):
        secrets.unlock("wrong password value")


def test_new_master_password_requires_eight_characters(tmp_path) -> None:
    secrets = EncryptedSQLiteSecretStore(tmp_path / "chat.sqlite3")

    with pytest.raises(InvalidMasterPasswordError, match="at least 8"):
        secrets.unlock("1234567", create_if_missing=True)

    secrets.unlock("12345678", create_if_missing=True)
    assert secrets.unlocked


def test_plaintext_secret_is_migrated_and_removed(tmp_path) -> None:
    database = tmp_path / "chat.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE chat_secrets (
                account TEXT PRIMARY KEY,
                api_key TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO chat_secrets VALUES (?, ?, ?)",
            ("primary", "legacy-plaintext-key", "now"),
        )

    secrets = EncryptedSQLiteSecretStore(database)
    secrets.unlock("migration password", create_if_missing=True)

    assert secrets.get("primary") == "legacy-plaintext-key"
    assert b"legacy-plaintext-key" not in database.read_bytes()
    with sqlite3.connect(database) as connection:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'chat_secrets'"
        ).fetchone()
    assert table is None


def test_missing_key_is_treated_as_incomplete_setup(tmp_path) -> None:
    configs = ChatConfigStore(tmp_path / "chat.sqlite3")
    configs.save(_config())

    assert ensure_chat_setup(configs, MemorySecretStore(), show_dialog=False) is None


def test_openai_compatible_client_builds_request(monkeypatch) -> None:
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "你好", "tool_calls": []}}]}
            ).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    model = OpenAICompatibleLanguageModel(_config(), "secret")

    response = model.complete(LLMRequest((ChatMessage("user", "hello"),)))

    assert response.text == "你好"
    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["authorization"] == "Bearer secret"
    assert captured["payload"]["model"] == "example-chat"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["timeout"] == 10.0
