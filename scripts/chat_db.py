from __future__ import annotations

import sqlite3
import uuid
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_DIR = PROJECT_ROOT / "data"
DB_PATH = DB_DIR / "chat_sessions.sqlite3"


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


def migrate_from_json(json_dir: Optional[Path] = None) -> None:
    """If the DB has no chats yet, import existing JSON chat files from `json_dir`.

    This runs silently and will not delete the original JSON files.
    """
    if json_dir is None:
        json_dir = PROJECT_ROOT / "chat_sessions"

    if not json_dir.exists() or not json_dir.is_dir():
        return

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(1) as c FROM chats")
    row = cur.fetchone()
    if row and row[0] > 0:
        conn.close()
        return

    for fname in os.listdir(json_dir):
        if not fname.endswith(".json"):
            continue
        path = json_dir / fname
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue

        cid = data.get("id") or str(uuid.uuid4())
        title = data.get("title") or "New Chat"
        updated_at = data.get("updated_at") or datetime.now().isoformat()
        messages = data.get("messages") or []

        cur.execute(
            "INSERT OR REPLACE INTO chats (id, title, updated_at) VALUES (?, ?, ?)",
            (cid, title, updated_at),
        )

        for m in messages:
            timestamp = m.get("timestamp") or updated_at
            cur.execute(
                "INSERT INTO messages (chat_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (cid, m.get("role"), m.get("content"), timestamp),
            )

    conn.commit()
    conn.close()


# Attempt to migrate existing JSON sessions into SQLite if DB is empty
migrate_from_json()
