"""
focus_tracker.py - Anti-procrastination module
Features: Todo list, Pomodoro timer, Daily check-in, Motivation quotes
"""
import random
import aiosqlite
from datetime import datetime
from database import DB_PATH

MOTIVATION_QUOTES = [
    "Bạn không cần cảm thấy sẵn sàng. Bạn chỉ cần bắt đầu.",
    "Việc khó nhất là bắt đầu. Sau đó mọi thứ sẽ dễ hơn.",
    "Làm ngay bây giờ, hoàn hảo sau.",
    "Mỗi bước nhỏ đều đưa bạn đến gần mục tiêu hơn.",
    "Đừng chờ động lực. Hành động tạo ra động lực.",
    "1% mỗi ngày = 37 lần tốt hơn sau 1 năm.",
    "Kẻ thù lớn nhất của bạn là phiên bản lười biếng của chính bạn.",
    "Làm xong còn hơn làm hoàn hảo.",
    "Hôm nay bạn làm gì để hối tiếc ít hơn ngày mai?",
    "Thành công không đến từ những gì bạn làm thỉnh thoảng, mà từ những gì bạn làm nhất quán.",
    "Não bạn nói 'chưa', nhưng tay bạn cứ làm đi.",
    "5 giây. Đếm ngược 5-4-3-2-1 rồi bắt đầu.",
    "Trì hoãn là kẻ cướp thời gian. Đừng để nó thắng hôm nay.",
    "Việc bạn né tránh nhất thường là việc bạn cần làm nhất.",
    "Bắt đầu tệ còn hơn không bắt đầu.",
]


async def init_focus_db():
    async with aiosqlite.connect(DB_PATH) as db:
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
        await db.commit()


# ── Tasks ─────────────────────────────────────────────────────────────────────

async def add_task(user_id: int, title: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO tasks (user_id, title, status, created_at) VALUES (?, ?, 'pending', ?)",
            (user_id, title, datetime.utcnow().isoformat())
        )
        await db.commit()
        return cursor.lastrowid


async def get_tasks(user_id: int, status: str = None) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if status:
            async with db.execute(
                "SELECT * FROM tasks WHERE user_id = ? AND status = ? ORDER BY id DESC",
                (user_id, status)
            ) as cursor:
                rows = await cursor.fetchall()
        else:
            async with db.execute(
                "SELECT * FROM tasks WHERE user_id = ? ORDER BY status ASC, id DESC",
                (user_id,)
            ) as cursor:
                rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def complete_task(user_id: int, task_id: int) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "UPDATE tasks SET status='done', done_at=? WHERE id=? AND user_id=? AND status='pending'",
            (datetime.utcnow().isoformat(), task_id, user_id)
        )
        await db.commit()
        return cursor.rowcount > 0


async def delete_task(user_id: int, task_id: int) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "DELETE FROM tasks WHERE id=? AND user_id=?",
            (task_id, user_id)
        )
        await db.commit()
        return cursor.rowcount > 0


async def clear_done_tasks(user_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM tasks WHERE user_id=? AND status='done'", (user_id,))
        await db.commit()


# ── Pomodoro ──────────────────────────────────────────────────────────────────

async def start_pomodoro(user_id: int, task_title: str = "") -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO pomodoro_sessions (user_id, task_title, started_at) VALUES (?, ?, ?)",
            (user_id, task_title, datetime.utcnow().isoformat())
        )
        await db.commit()
        return cursor.lastrowid


async def get_pomodoro_count_today(user_id: int) -> int:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM pomodoro_sessions WHERE user_id=? AND started_at >= ?",
            (user_id, today)
        ) as cursor:
            row = await cursor.fetchone()
    return row[0] if row else 0


# ── Text builders ─────────────────────────────────────────────────────────────

async def build_task_list(user_id: int) -> str:
    tasks = await get_tasks(user_id)
    if not tasks:
        return "📋 <b>Danh sách việc cần làm</b>\n\nChưa có việc nào. Dùng /todo để thêm!"

    pending = [t for t in tasks if t["status"] == "pending"]
    done = [t for t in tasks if t["status"] == "done"]

    text = "📋 <b>Danh sách việc cần làm</b>\n\n"

    if pending:
        text += f"⏳ <b>Chưa xong ({len(pending)}):</b>\n"
        for t in pending:
            text += f"  <code>[{t['id']}]</code> {t['title']}\n"
        text += "\n"

    if done:
        text += f"✅ <b>Đã xong ({len(done)}):</b>\n"
        for t in done[-5:]:  # show last 5 done
            text += f"  <s>{t['title']}</s>\n"

    text += "\n<i>Dùng /done [số] để đánh dấu hoàn thành\nDùng /deltask [số] để xóa</i>"
    return text


async def get_daily_summary(user_id: int) -> str:
    tasks = await get_tasks(user_id)
    pending = [t for t in tasks if t["status"] == "pending"]
    done_today = [
        t for t in tasks
        if t["status"] == "done" and t.get("done_at", "")[:10] == datetime.utcnow().strftime("%Y-%m-%d")
    ]
    pomodoros = await get_pomodoro_count_today(user_id)

    text = "📊 <b>Tổng kết hôm nay</b>\n"
    text += "━━━━━━━━━━━━━━━━━━━━\n"
    text += f"✅ Việc hoàn thành: <b>{len(done_today)}</b>\n"
    text += f"⏳ Việc còn lại: <b>{len(pending)}</b>\n"
    text += f"🍅 Pomodoro: <b>{pomodoros}</b> phiên\n"
    text += "━━━━━━━━━━━━━━━━━━━━\n"

    if done_today:
        text += "\n<b>Đã làm xong:</b>\n"
        for t in done_today:
            text += f"  ✅ {t['title']}\n"

    if pending:
        text += "\n<b>Chưa xong:</b>\n"
        for t in pending[:5]:
            text += f"  ⏳ {t['title']}\n"

    if pomodoros >= 4:
        text += "\n🏆 Tuyệt vời! Bạn đã làm việc rất chăm chỉ hôm nay!"
    elif pomodoros >= 2:
        text += "\n👍 Tốt lắm! Tiếp tục duy trì nhé!"
    elif pomodoros == 0 and len(done_today) == 0:
        text += "\n💪 Ngày mai bắt đầu sớm hơn nhé!"

    return text


def get_motivation() -> str:
    quote = random.choice(MOTIVATION_QUOTES)
    return f'💪 <b>Câu động lực hôm nay:</b>\n\n<i>"{quote}"</i>'
