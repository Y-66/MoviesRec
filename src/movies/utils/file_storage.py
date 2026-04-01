from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict


_SESSION_ID_PATTERN = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_session_id(session_id: str) -> str:
    candidate = (session_id or "default").strip()
    if not candidate:
        candidate = "default"
    return _SESSION_ID_PATTERN.sub("_", candidate)


def history_root(base_dir: Path | None = None) -> Path:
    root = (base_dir or Path.cwd()) / "chat_history"
    root.mkdir(parents=True, exist_ok=True)
    return root


def session_history_path(session_id: str, base_dir: Path | None = None) -> Path:
    return history_root(base_dir) / f"{_sanitize_session_id(session_id)}.json"


def load_session_history(session_id: str, base_dir: Path | None = None) -> list[BaseMessage]:
    path = session_history_path(session_id, base_dir)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_messages = payload.get("messages", [])
    if not isinstance(raw_messages, list):
        return []
    return list(messages_from_dict(raw_messages))


def save_session_history(
    session_id: str,
    messages: list[BaseMessage],
    base_dir: Path | None = None,
) -> Path:
    path = session_history_path(session_id, base_dir)
    payload = {
        "session_id": _sanitize_session_id(session_id),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "message_count": len(messages),
        "messages": messages_to_dict(messages),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def read_session_history_payload(session_id: str, base_dir: Path | None = None) -> dict[str, Any]:
    path = session_history_path(session_id, base_dir)
    if not path.exists():
        return {
            "session_id": _sanitize_session_id(session_id),
            "updated_at": None,
            "message_count": 0,
            "messages": [],
        }

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return {
        "session_id": payload.get("session_id", _sanitize_session_id(session_id)),
        "updated_at": payload.get("updated_at"),
        "message_count": payload.get("message_count", 0),
        "messages": payload.get("messages", []),
    }


def list_session_summaries(base_dir: Path | None = None) -> list[dict[str, Any]]:
    root = history_root(base_dir)
    sessions: list[dict[str, Any]] = []

    for file_path in root.glob("*.json"):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            sessions.append(
                {
                    "session_id": payload.get("session_id", file_path.stem),
                    "updated_at": payload.get("updated_at"),
                    "message_count": payload.get("message_count", 0),
                    "file": str(file_path.name),
                }
            )
        except Exception:
            continue

    sessions.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return sessions