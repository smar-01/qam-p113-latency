from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def load_key() -> str:
    """Read only the project-local key; never log its value."""
    key = os.environ.get("DATABENTO_API_KEY", "").strip()
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if re.match(r"^\s*(?:export\s+)?DATABENTO_API_KEY\s*=", line):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise ValueError("No key configured. Run python scripts/setup_key.py to open the masked local prompt.")
    return key


def safe_error(exc: Exception) -> str:
    text = str(exc)
    try:
        key = load_key()
        for token in (key, base64.b64encode((key + ":").encode()).decode()):
            text = text.replace(token, "[REDACTED]")
    except ValueError:
        pass
    text = re.sub(r"db-[A-Za-z0-9_-]+", "[REDACTED]", text)
    return f"{type(exc).__name__}: {text[:2000]}"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def read_request(path: Path) -> dict:
    import pandas as pd

    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    keys = {"dataset", "schema", "symbols", "start", "end", "limit"}
    if set(raw) != keys:
        raise ValueError(f"Pilot configuration requires exactly: {', '.join(sorted(keys))}")
    if raw["schema"] not in {"trades", "mbp-1"}:
        raise ValueError("First experiment supports trades or mbp-1; MBO needs a separate order-book reconstruction.")
    if not isinstance(raw["symbols"], list) or not 1 <= len(raw["symbols"]) <= 10:
        raise ValueError("Pilot requires 1–10 explicit symbols.")
    if any(not isinstance(s, str) or not s.strip() or s.strip().upper() == "ALL_SYMBOLS" or s != s.strip() or "," in s for s in raw["symbols"]):
        raise ValueError("Use explicit nonempty ticker symbols; ALL_SYMBOLS is disabled.")
    start, end = pd.Timestamp(raw["start"]), pd.Timestamp(raw["end"])
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Request times must have explicit timezones.")
    if not pd.Timedelta(0) < end - start <= pd.Timedelta(minutes=60):
        raise ValueError("Pilot requests must be between zero and 60 minutes.")
    if type(raw["limit"]) is not int or not 1 <= raw["limit"] <= 1000000:
        raise ValueError("Pilot requires a record limit from 1 to 1,000,000.")
    return dict(raw, stype_in="raw_symbol")
