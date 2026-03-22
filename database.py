"""
database.py - Async SQLite layer using aiosqlite
"""
import aiosqlite
import asyncio
from datetime import datetime
from typing import Any, Optional

DB_PATH = "bot_data.db"


async def init_db():
    """Initialize all database tables."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                PRIMARY KEY (user_id, key)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS summary (
                user_id INTEGER PRIMARY KEY,
                summary_text TEXT,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                message TEXT NOT NULL,
                fire_at TEXT NOT NULL,
                repeat TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                user_id INTEGER PRIMARY KEY,
                profile_text TEXT,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS rag_docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS money_tracker (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                amount REAL NOT NULL,
                note TEXT,
                category TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                priority INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                done_at TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS pomodoro_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                task_title TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_tasks_user ON tasks(user_id)")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS allowed_users (
                user_id INTEGER PRIMARY KEY,
                added_by INTEGER NOT NULL,
                added_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        # Indexes for performance
        await db.execute("CREATE INDEX IF NOT EXISTS idx_history_user ON history(user_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_reminders_fire ON reminders(fire_at)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_token_user ON token_usage(user_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_money_user ON money_tracker(user_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_rag_user ON rag_docs(user_id)")

        await db.commit()


# ── History ──────────────────────────────────────────────────────────────────

async def add_message(user_id: int, role: str, content: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO history (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (user_id, role, content, datetime.utcnow().isoformat())
        )
        await db.commit()


async def get_history(user_id: int, limit: int = 20) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT role, content, created_at FROM history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit)
        ) as cursor:
            rows = await cursor.fetchall()
    # Return in chronological order
    return [{"role": r["role"], "content": r["content"], "created_at": r["created_at"]} for r in reversed(rows)]


async def clear_history(user_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
        await db.commit()


# ── Settings ─────────────────────────────────────────────────────────────────

async def get_setting(user_id: int, key: str, default: Any = None) -> Any:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT value FROM user_settings WHERE user_id = ? AND key = ?",
            (user_id, key)
        ) as cursor:
            row = await cursor.fetchone()
    return row[0] if row else default


async def set_setting(user_id: int, key: str, value: Any):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO user_settings (user_id, key, value) VALUES (?, ?, ?)",
            (user_id, key, str(value) if value is not None else None)
        )
        await db.commit()


# ── Summary ───────────────────────────────────────────────────────────────────

async def get_summary(user_id: int) -> Optional[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT summary_text FROM summary WHERE user_id = ?",
            (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
    return row[0] if row else None


async def set_summary(user_id: int, text: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO summary (user_id, summary_text, updated_at) VALUES (?, ?, ?)",
            (user_id, text, datetime.utcnow().isoformat())
        )
        await db.commit()


# ── Reminders ────────────────────────────────────────────────────────────────

async def add_reminder(user_id: int, message: str, fire_at: str, repeat: Optional[str] = None) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO reminders (user_id, message, fire_at, repeat, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, message, fire_at, repeat, datetime.utcnow().isoformat())
        )
        await db.commit()
        return cursor.lastrowid


async def get_pending_reminders() -> list[dict]:
    """Returns reminders where fire_at <= now."""
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM reminders WHERE fire_at <= ?",
            (now,)
        ) as cursor:
            rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def update_reminder_fire_at(reminder_id: int, fire_at: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE reminders SET fire_at = ? WHERE id = ?",
            (fire_at, reminder_id)
        )
        await db.commit()


async def delete_reminder(reminder_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
        await db.commit()


async def get_user_reminders(user_id: int) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM reminders WHERE user_id = ? ORDER BY fire_at ASC",
            (user_id,)
        ) as cursor:
            rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ── Profile ───────────────────────────────────────────────────────────────────

async def get_profile(user_id: int) -> Optional[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT profile_text FROM user_profile WHERE user_id = ?",
            (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
    return row[0] if row else None


async def set_profile(user_id: int, text: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO user_profile (user_id, profile_text, updated_at) VALUES (?, ?, ?)",
            (user_id, text, datetime.utcnow().isoformat())
        )
        await db.commit()


# ── Token Usage ───────────────────────────────────────────────────────────────

async def log_token_usage(user_id: int, model: str, prompt_tokens: int, completion_tokens: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO token_usage (user_id, model, prompt_tokens, completion_tokens, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, model, prompt_tokens, completion_tokens, datetime.utcnow().isoformat())
        )
        await db.commit()


async def get_token_report(user_id: int) -> list[dict]:
    """Returns per-model token usage aggregated for the user."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT model,
                      SUM(prompt_tokens) as total_prompt,
                      SUM(completion_tokens) as total_completion,
                      COUNT(*) as call_count
               FROM token_usage
               WHERE user_id = ?
               GROUP BY model
               ORDER BY model""",
            (user_id,)
        ) as cursor:
            rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ── Money Tracker ─────────────────────────────────────────────────────────────

async def add_money_entry(user_id: int, amount: float, note: str, category: str = "") -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO money_tracker (user_id, amount, note, category, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, amount, note, category, datetime.utcnow().isoformat())
        )
        await db.commit()
        return cursor.lastrowid


async def get_money_entries(user_id: int, since_iso: Optional[str] = None) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if since_iso:
            async with db.execute(
                "SELECT * FROM money_tracker WHERE user_id = ? AND created_at >= ? ORDER BY created_at DESC",
                (user_id, since_iso)
            ) as cursor:
                rows = await cursor.fetchall()
        else:
            async with db.execute(
                "SELECT * FROM money_tracker WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,)
            ) as cursor:
                rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def delete_money_entry(entry_id: int, user_id: int) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "DELETE FROM money_tracker WHERE id = ? AND user_id = ?",
            (entry_id, user_id)
        )
        await db.commit()
        return cursor.rowcount > 0


# ── RAG Docs ──────────────────────────────────────────────────────────────────

async def add_rag_chunk(user_id: int, filename: str, chunk_index: int, content: str, embedding: str = ""):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO rag_docs (user_id, filename, chunk_index, content, embedding, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, filename, chunk_index, content, embedding, datetime.utcnow().isoformat())
        )
        await db.commit()


async def get_rag_chunks(user_id: int) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM rag_docs WHERE user_id = ? ORDER BY filename, chunk_index",
            (user_id,)
        ) as cursor:
            rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def count_user_docs(user_id: int) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(DISTINCT filename) FROM rag_docs WHERE user_id = ?",
            (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
    return row[0] if row else 0


async def delete_rag_doc(user_id: int, filename: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "DELETE FROM rag_docs WHERE user_id = ? AND filename = ?",
            (user_id, filename)
        )
        await db.commit()
        return cursor.rowcount > 0


async def list_rag_docs(user_id: int) -> list[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT DISTINCT filename FROM rag_docs WHERE user_id = ? ORDER BY filename",
            (user_id,)
        ) as cursor:
            rows = await cursor.fetchall()
    return [r[0] for r in rows]


# ── Allowed Users (whitelist) ─────────────────────────────────────────────────

async def add_allowed_user(user_id: int, added_by: int) -> bool:
    """Add user to whitelist. Returns True if newly added, False if already exists."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT OR IGNORE INTO allowed_users (user_id, added_by, added_at) VALUES (?, ?, ?)",
            (user_id, added_by, datetime.utcnow().isoformat())
        )
        await db.commit()
        return cursor.rowcount > 0


async def remove_allowed_user(user_id: int) -> bool:
    """Remove user from whitelist. Returns True if removed."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "DELETE FROM allowed_users WHERE user_id = ?", (user_id,)
        )
        await db.commit()
        return cursor.rowcount > 0


async def is_user_allowed(user_id: int) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT 1 FROM allowed_users WHERE user_id = ?", (user_id,)
        ) as cursor:
            return await cursor.fetchone() is not None


async def list_allowed_users() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT user_id, added_by, added_at FROM allowed_users ORDER BY added_at"
        ) as cursor:
            rows = await cursor.fetchall()
    return [dict(r) for r in rows]
