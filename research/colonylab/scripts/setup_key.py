"""Local masked prompt. Never prints or sends the entered key to the agent."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tkinter as tk
from tkinter import messagebox, simpledialog

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    status = {"status": "cancelled"}
    try:
        while True:
            key = simpledialog.askstring(
                "Databento API key — local setup",
                "Paste your Databento API key below.\n\n"
                "It will be stored in this project's ignored .env file.\n"
                "The key is masked and will not be sent to the chat.\n"
                "This step does not download or purchase data.",
                show="*", parent=root,
            )
            if key is None:
                return
            key = key.strip()
            if not re.fullmatch(r"db-[A-Za-z0-9_-]{10,}", key):
                messagebox.showerror("Check the key", "Expected a Databento key beginning with db-.", parent=root)
                continue
            env = ROOT / ".env"
            lines = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
            lines = [line for line in lines if not re.match(r"^\s*(?:export\s+)?DATABENTO_API_KEY\s*=", line)]
            lines.append("DATABENTO_API_KEY=" + key)
            temp = ROOT / ".env.key-setup.tmp"
            try:
                temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
                os.replace(temp, env)
            finally:
                if temp.exists():
                    temp.unlink()
            status = {"status": "saved", "variable": "DATABENTO_API_KEY", "destination": ".env"}
            messagebox.showinfo("Key saved", "The key is saved locally. You can close this prompt.\nThe research setup will now check metadata.", parent=root)
            return
    except Exception:
        status = {"status": "error", "message": "Local key setup failed; no secret is included in this status."}
        messagebox.showerror("Setup failed", "Could not save the key locally. Please try again.", parent=root)
    finally:
        (ROOT / ".key-setup-status.json").write_text(json.dumps(status), encoding="utf-8")
        root.destroy()


if __name__ == "__main__":
    main()
