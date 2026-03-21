"""
reminder_system.py - NLP reminder parser + background loop
Supports Vietnamese time expressions: "7h tối", "ngày mai", "mỗi ngày", etc.
"""
import asyncio
import logging
import os
import json
import re
from datetime import datetime, timedelta
from typing import Optional

from database import (
    add_reminder, get_pending_reminders, update_reminder_fire_at,
    delete_reminder, get_user_reminders
)

logger = logging.getLogger(__name__)

# Repeat interval in seconds
REPEAT_INTERVALS = {
    "mỗi phút":    60,
    "mỗi giờ":     3600,
    "mỗi ngày":    86400,
    "hằng ngày":   86400,
    "daily":       86400,
    "mỗi tuần":    7 * 86400,
    "hàng tuần":   7 * 86400,
    "weekly":      7 * 86400,
}

HOUR_KEYWORDS = {
    "sáng":  0,
    "trưa":  12,
    "chiều": 12,  # afternoon starts at noon
    "tối":   12,  # evening offset added in parsing
    "đêm":   12,
}


def _now_utc() -> datetime:
    return datetime.utcnow()


def _parse_simple_time(text: str) -> Optional[datetime]:
    """
    Try basic pattern matching for common Vietnamese time expressions.
    Returns UTC datetime or None if not parseable.
    """
    text_lower = text.lower().strip()
    now = _now_utc()

    # Pattern: "sau N phút/giờ"
    after_match = re.search(r'sau\s+(\d+)\s*(phút|giờ|tiếng)', text_lower)
    if after_match:
        amount = int(after_match.group(1))
        unit = after_match.group(2)
        if unit == "phút":
            return now + timedelta(minutes=amount)
        else:
            return now + timedelta(hours=amount)

    # Pattern: "Nh giờ [sáng/trưa/chiều/tối]"
    hour_match = re.search(r'(\d{1,2})[hH:]\s*(\d{0,2})\s*(sáng|trưa|chiều|tối|đêm)?', text_lower)
    if hour_match:
        hour = int(hour_match.group(1))
        minute = int(hour_match.group(2)) if hour_match.group(2) else 0
        period = hour_match.group(3)

        if period in ("chiều", "tối", "đêm") and hour < 12:
            hour += 12
        elif period == "trưa" and hour < 12:
            hour = 12

        # Build target datetime for today
        target = now.replace(hour=hour % 24, minute=minute, second=0, microsecond=0)

        # Check for "ngày mai"
        if "ngày mai" in text_lower or "tomorrow" in text_lower:
            target += timedelta(days=1)
        elif target <= now:
            # Time already passed today → schedule for tomorrow
            target += timedelta(days=1)

        return target

    # "ngày mai" alone → tomorrow same time + 1 hour
    if "ngày mai" in text_lower:
        return now + timedelta(days=1)

    # "1 tiếng nữa" / "30 phút nữa"
    nua_match = re.search(r'(\d+)\s*(tiếng|giờ|phút)\s*nữa', text_lower)
    if nua_match:
        amount = int(nua_match.group(1))
        unit = nua_match.group(2)
        if unit == "phút":
            return now + timedelta(minutes=amount)
        else:
            return now + timedelta(hours=amount)

    return None


def _detect_repeat(text: str) -> Optional[str]:
    """Returns repeat string like 'daily', 'weekly', or an interval string like '86400'."""
    text_lower = text.lower()
    for keyword, seconds in REPEAT_INTERVALS.items():
        if keyword in text_lower:
            return str(seconds)
    return None


async def parse_reminder_nlp(text: str, user_id: int) -> dict:
    """
    Use Mistral to extract reminder details from natural language.
    Returns dict with keys: message, fire_at (ISO string), repeat (optional string).
    Falls back to simple parsing if Mistral unavailable.
    """
    # Try simple parsing first
    fire_at_dt = _parse_simple_time(text)
    repeat = _detect_repeat(text)

    # Extract reminder message (remove time parts)
    message = re.sub(
        r'(nhắc|remind|reminder|lúc|vào|sau|ngày mai|mỗi|hằng|hàng|'
        r'\d+[hH:]\d*\s*(?:sáng|trưa|chiều|tối|đêm)?|'
        r'sau\s+\d+\s*(?:phút|giờ|tiếng)|'
        r'\d+\s*(?:phút|giờ|tiếng)\s*nữa)',
        '',
        text,
        flags=re.IGNORECASE
    ).strip()
    message = re.sub(r'\s+', ' ', message).strip(" ,.")

    if not message:
        message = text  # fallback to full text

    if fire_at_dt:
        return {
            "message": message,
            "fire_at": fire_at_dt.isoformat(),
            "repeat": repeat,
        }

    # Use Mistral NLP as fallback
    try:
        from llm_core import _call_mistral_sync
        now_str = _now_utc().strftime("%Y-%m-%d %H:%M")
        system = (
            "Trích xuất thông tin nhắc nhở từ văn bản. "
            f"Thời gian hiện tại: {now_str} UTC. "
            "Trả về JSON: {\"message\": \"...\", \"fire_at\": \"YYYY-MM-DDTHH:MM:SS\", \"repeat\": null hoặc \"86400\" hoặc \"604800\"}"
        )
        raw = await _call_mistral_sync(system, text)
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return {
                "message": data.get("message", text),
                "fire_at": data.get("fire_at", (_now_utc() + timedelta(hours=1)).isoformat()),
                "repeat": data.get("repeat"),
            }
    except Exception as e:
        logger.warning(f"Mistral NLP parse failed: {e}")

    # Final fallback: 1 hour from now
    return {
        "message": message or text,
        "fire_at": (_now_utc() + timedelta(hours=1)).isoformat(),
        "repeat": repeat,
    }


async def set_reminder_from_text(user_id: int, text: str) -> str:
    """Parse text and store reminder. Returns confirmation message."""
    parsed = await parse_reminder_nlp(text, user_id)
    reminder_id = await add_reminder(
        user_id,
        parsed["message"],
        parsed["fire_at"],
        parsed.get("repeat")
    )

    try:
        fire_dt = datetime.fromisoformat(parsed["fire_at"])
        time_str = fire_dt.strftime("%H:%M %d/%m/%Y")
    except Exception:
        time_str = parsed["fire_at"]

    repeat_str = ""
    if parsed.get("repeat"):
        seconds = int(parsed["repeat"])
        if seconds == 86400:
            repeat_str = " (lặp lại mỗi ngày)"
        elif seconds == 604800:
            repeat_str = " (lặp lại mỗi tuần)"
        elif seconds == 3600:
            repeat_str = " (lặp lại mỗi giờ)"

    return (
        f"⏰ Đã đặt nhắc nhở:\n"
        f"📝 <b>{parsed['message']}</b>\n"
        f"🕐 Lúc: <code>{time_str} UTC</code>{repeat_str}\n"
        f"ID: <code>#{reminder_id}</code>"
    )


async def list_reminders_text(user_id: int) -> str:
    """Format user's reminders as HTML."""
    reminders = await get_user_reminders(user_id)
    if not reminders:
        return "📭 Bạn chưa có nhắc nhở nào.\n\nDùng <code>/remind &lt;nội dung&gt;</code> để thêm."

    lines = ["⏰ <b>Danh sách nhắc nhở của bạn:</b>\n"]
    for r in reminders:
        try:
            fire_dt = datetime.fromisoformat(r["fire_at"])
            time_str = fire_dt.strftime("%H:%M %d/%m/%Y")
        except Exception:
            time_str = r["fire_at"]

        repeat_badge = " 🔁" if r.get("repeat") else ""
        lines.append(
            f"<code>[#{r['id']}]</code> {r['message']}\n"
            f"  ⏱ <i>{time_str} UTC</i>{repeat_badge}"
        )

    return "\n\n".join(lines)


async def reminder_loop(bot):
    """
    Background task: checks every 30 seconds for pending reminders
    and sends Telegram messages.
    """
    logger.info("Reminder loop started.")
    while True:
        try:
            pending = await get_pending_reminders()
            for reminder in pending:
                try:
                    await bot.send_message(
                        chat_id=reminder["user_id"],
                        text=f"⏰ <b>Nhắc nhở!</b>\n\n{reminder['message']}",
                        parse_mode="HTML"
                    )
                    logger.info(f"Reminder #{reminder['id']} fired for user {reminder['user_id']}")
                except Exception as e:
                    logger.error(f"Failed to send reminder #{reminder['id']}: {e}")

                # Handle repeat
                if reminder.get("repeat"):
                    try:
                        interval_sec = int(reminder["repeat"])
                        current_fire = datetime.fromisoformat(reminder["fire_at"])
                        next_fire = current_fire + timedelta(seconds=interval_sec)
                        # Ensure next_fire is in the future
                        now = _now_utc()
                        while next_fire <= now:
                            next_fire += timedelta(seconds=interval_sec)
                        await update_reminder_fire_at(reminder["id"], next_fire.isoformat())
                    except Exception as e:
                        logger.error(f"Failed to reschedule reminder #{reminder['id']}: {e}")
                        await delete_reminder(reminder["id"])
                else:
                    await delete_reminder(reminder["id"])

        except Exception as e:
            logger.error(f"Reminder loop error: {e}")

        await asyncio.sleep(30)
