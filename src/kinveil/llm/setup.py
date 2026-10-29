from __future__ import annotations

import getpass
from dataclasses import dataclass

from .config import (
    ChatConfigStore,
    ChatModelConfig,
    EncryptedSQLiteSecretStore,
    InvalidMasterPasswordError,
    SecretStore,
)


@dataclass(frozen=True)
class ChatSetupResult:
    config: ChatModelConfig
    api_key: str


def unlock_chat_secrets(
    secrets: EncryptedSQLiteSecretStore,
    *,
    show_dialog: bool,
) -> bool:
    if secrets.unlocked:
        return True
    if not show_dialog:
        return False
    try:
        return _show_master_password_dialog(secrets)
    except ChatSetupUIUnavailableError:
        return _prompt_master_password(secrets)


class ChatSetupUIUnavailableError(RuntimeError):
    pass


def ensure_chat_setup(
    configs: ChatConfigStore,
    secrets: SecretStore,
    *,
    show_dialog: bool,
    force_dialog: bool = False,
) -> ChatSetupResult | None:
    active = configs.active()
    if active is not None and not force_dialog:
        key = secrets.get(active.id)
        if key:
            return ChatSetupResult(active, key)
    if not show_dialog:
        return None
    try:
        entered = _show_setup_dialog(active)
    except ChatSetupUIUnavailableError:
        entered = _prompt_setup(active)
    if entered is None:
        return None
    config, key = entered
    secrets.set(config.id, key)
    configs.save(config)
    return ChatSetupResult(config, key)


def _show_setup_dialog(
    existing: ChatModelConfig | None,
) -> tuple[ChatModelConfig, str] | None:
    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError as exc:
        raise ChatSetupUIUnavailableError("Python Tk support is unavailable") from exc

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise ChatSetupUIUnavailableError("a graphical display is unavailable") from exc
    root.title("Kinveil Chat Model")
    root.resizable(False, False)
    root.attributes("-topmost", True)
    values = {
        "name": tk.StringVar(value=existing.name if existing else "My chat model"),
        "url": tk.StringVar(value=existing.base_url if existing else "https://api.deepseek.com"),
        "model": tk.StringVar(value=existing.model if existing else "deepseek-chat"),
        "key": tk.StringVar(),
    }
    fields = (
        ("Display name", "name", False),
        ("API base URL", "url", False),
        ("Model", "model", False),
        ("API key", "key", True),
    )
    for row, (label, key, secret) in enumerate(fields):
        tk.Label(root, text=label, anchor="w", width=14).grid(
            row=row, column=0, padx=(16, 8), pady=7, sticky="w"
        )
        tk.Entry(root, textvariable=values[key], width=48, show="•" if secret else "").grid(
            row=row, column=1, padx=(0, 16), pady=7
        )
    result: list[tuple[ChatModelConfig, str]] = []

    def save() -> None:
        try:
            config = ChatModelConfig(
                id=existing.id if existing else "default",
                name=values["name"].get().strip(),
                base_url=values["url"].get().strip(),
                model=values["model"].get().strip(),
            )
            api_key = values["key"].get().strip()
            if not api_key:
                raise ValueError("API key cannot be empty")
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc), parent=root)
            return
        result.append((config, api_key))
        root.destroy()

    buttons = tk.Frame(root)
    buttons.grid(row=len(fields), column=0, columnspan=2, pady=(8, 16))
    tk.Button(buttons, text="Cancel", command=root.destroy, width=10).pack(side="left", padx=6)
    tk.Button(buttons, text="Save", command=save, width=10, default="active").pack(
        side="left", padx=6
    )
    root.bind("<Return>", lambda _event: save())
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.eval("tk::PlaceWindow . center")
    root.mainloop()
    return result[0] if result else None


def _prompt_setup(
    existing: ChatModelConfig | None,
) -> tuple[ChatModelConfig, str] | None:
    defaults = existing or ChatModelConfig(
        id="default",
        name="My chat model",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
    )

    def ask(label: str, default: str) -> str:
        entered = input(f"{label} [{default}]: ").strip()
        return entered or default

    try:
        config = ChatModelConfig(
            id=defaults.id,
            name=ask("Display name", defaults.name),
            base_url=ask("API base URL", defaults.base_url),
            model=ask("Model", defaults.model),
        )
        api_key = getpass.getpass("API key: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not api_key:
        return None
    return config, api_key


def _show_master_password_dialog(secrets: EncryptedSQLiteSecretStore) -> bool:
    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError as exc:
        raise ChatSetupUIUnavailableError("Python Tk support is unavailable") from exc

    creating = not secrets.initialized
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise ChatSetupUIUnavailableError("a graphical display is unavailable") from exc
    root.title("Create Kinveil Master Password" if creating else "Unlock Kinveil")
    root.resizable(False, False)
    root.attributes("-topmost", True)
    password = tk.StringVar()
    confirmation = tk.StringVar()
    description = (
        "Create a master password to encrypt the local API key.\n"
        "This password is not stored and cannot be recovered."
        if creating
        else "Enter the master password to decrypt the local API key in memory."
    )
    tk.Label(root, text=description, justify="left").grid(
        row=0, column=0, columnspan=2, padx=18, pady=(16, 10), sticky="w"
    )
    tk.Label(root, text="Master password", width=16, anchor="w").grid(
        row=1, column=0, padx=(18, 8), pady=7, sticky="w"
    )
    password_entry = tk.Entry(root, textvariable=password, width=38, show="•")
    password_entry.grid(row=1, column=1, padx=(0, 18), pady=7)
    next_row = 2
    if creating:
        tk.Label(root, text="Confirm password", width=16, anchor="w").grid(
            row=2, column=0, padx=(18, 8), pady=7, sticky="w"
        )
        tk.Entry(root, textvariable=confirmation, width=38, show="•").grid(
            row=2, column=1, padx=(0, 18), pady=7
        )
        next_row = 3
    unlocked: list[bool] = []

    def submit() -> None:
        entered = password.get()
        if creating and entered != confirmation.get():
            messagebox.showerror(
                "Passwords do not match",
                "Enter the same password twice.",
                parent=root,
            )
            return
        if creating and len(entered) < 8:
            messagebox.showerror(
                "Password too short",
                "Use at least 8 characters.",
                parent=root,
            )
            return
        try:
            secrets.unlock(entered, create_if_missing=creating)
        except InvalidMasterPasswordError as exc:
            password.set("")
            confirmation.set("")
            messagebox.showerror("Could not unlock", str(exc), parent=root)
            password_entry.focus_set()
            return
        password.set("")
        confirmation.set("")
        unlocked.append(True)
        root.destroy()

    buttons = tk.Frame(root)
    buttons.grid(row=next_row, column=0, columnspan=2, pady=(9, 16))
    tk.Button(buttons, text="Cancel", command=root.destroy, width=10).pack(side="left", padx=6)
    tk.Button(buttons, text="Create" if creating else "Unlock", command=submit, width=10).pack(
        side="left", padx=6
    )
    root.bind("<Return>", lambda _event: submit())
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.eval("tk::PlaceWindow . center")
    password_entry.focus_set()
    root.mainloop()
    return bool(unlocked)


def _prompt_master_password(secrets: EncryptedSQLiteSecretStore) -> bool:
    creating = not secrets.initialized
    for _attempt in range(3):
        entered = getpass.getpass(
            "Create Kinveil master password: " if creating else "Kinveil master password: "
        )
        if not entered:
            return False
        if creating:
            if len(entered) < 8:
                print("Master password must contain at least 8 characters.")
                continue
            confirmation = getpass.getpass("Confirm master password: ")
            if entered != confirmation:
                print("Master passwords do not match.")
                continue
        try:
            secrets.unlock(entered, create_if_missing=creating)
        except InvalidMasterPasswordError as exc:
            print(str(exc))
            continue
        return True
    return False
