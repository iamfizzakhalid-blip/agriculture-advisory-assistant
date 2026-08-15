from __future__ import annotations

import sqlite3
import uuid
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_DIR = PROJECT_ROOT / "data"
DB_PATH = DB_DIR / "chat_sessions.sqlite3"
LOGS_DIR = PROJECT_ROOT / "logs"


def _ensure_db_dir():
    DB_DIR.mkdir(parents=True, exist_ok=True)


def get_conn():
    _ensure_db_dir()
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS chats (
        id TEXT PRIMARY KEY,
        title TEXT,
        updated_at TEXT
    )
    """
    )
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT,
        role TEXT,
        content TEXT,
        timestamp TEXT,
        FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
    )
    """
    )
    conn.commit()
    conn.close()


def _safe_log_title(title: str) -> str:
    raw = (title or "New Chat").strip()
    raw = raw.replace("\\", " ").replace("/", " ")
    raw = raw.replace(":", " ").replace("*", " ").replace("?", " ")
    raw = raw.replace('"', " ").replace("<", " ").replace(">", " ")
    raw = raw.replace("|", " ")
    raw = re.sub(r"\s+", " ", raw).strip(" .")
    raw = raw.replace("..", ".")
    return raw or "Chat"


def ensure_chat_log_file(chat_id: str, title: str = "New Chat") -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    safe_title = _safe_log_title(title)
    suffix = str(chat_id)[:8]
    candidate = LOGS_DIR / f"{safe_title}_{suffix}.txt"
    if not candidate.exists():
        candidate.touch(exist_ok=True)
    return candidate


def append_chat_exchange(chat_id: str, title: str, user_query: str, assistant_response: str) -> Optional[Path]:
    try:
        log_path = ensure_chat_log_file(chat_id, title)
        entry = (
            "User:\n"
            f"{user_query.strip()}\n\n"
            "Assistant:\n"
            f"{assistant_response.strip()}\n\n"
            "--------------------------------------------------\n\n"
        )
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(entry)
        return log_path
    except Exception:
        return None


def create_chat(chat_id: Optional[str] = None, title: str = "New Chat") -> str:
    init_db()
    cid = chat_id or str(uuid.uuid4())
    now = datetime.now().isoformat()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO chats (id, title, updated_at) VALUES (?, ?, ?)",
        (cid, title, now),
    )
    conn.commit()
    conn.close()

    # Create the per-chat human-readable log file as soon as the chat is created.
    try:
        ensure_chat_log_file(cid, title)
    except Exception:
        pass

    return cid


def save_chat(chat_id: str, title: str, messages: List[Dict]) -> None:
    """Replace existing messages for a chat and update metadata."""
    init_db()
    now = datetime.now().isoformat()
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "INSERT OR REPLACE INTO chats (id, title, updated_at) VALUES (?, ?, ?)",
        (chat_id, title, now),
    )

    # Remove existing messages for this chat and insert fresh ones
    cur.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))

    for m in messages:
        timestamp = m.get("timestamp") or now
        cur.execute(
            "INSERT INTO messages (chat_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (chat_id, m.get("role"), m.get("content"), timestamp),
        )

    conn.commit()
    conn.close()


def load_chat(chat_id: str) -> Optional[Dict]:
    init_db()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, title, updated_at FROM chats WHERE id = ?", (chat_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return None

    cur.execute(
        "SELECT role, content, timestamp FROM messages WHERE chat_id = ? ORDER BY id ASC",
        (chat_id,),
    )
    msgs = [dict(role=r[0], content=r[1], timestamp=r[2]) for r in cur.fetchall()]
    conn.close()
    return {"id": row["id"], "title": row["title"], "updated_at": row["updated_at"], "messages": msgs}


def delete_chat(chat_id: str) -> None:
    init_db()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    cur.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()


def list_chats() -> List[Dict]:
    init_db()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, title, updated_at FROM chats ORDER BY updated_at DESC")
    chats = []
    rows = cur.fetchall()
    for row in rows:
        cid = row["id"]
        cur.execute(
            "SELECT role, content, timestamp FROM messages WHERE chat_id = ? ORDER BY id ASC",
            (cid,),
        )
        msgs = [dict(role=r[0], content=r[1], timestamp=r[2]) for r in cur.fetchall()]
        chats.append({"id": cid, "title": row["title"], "updated_at": row["updated_at"], "messages": msgs})

    conn.close()
    return chats


# Ensure DB exists on import
init_db()
