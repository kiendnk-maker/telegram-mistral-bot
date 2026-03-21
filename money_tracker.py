"""
money_tracker.py - Personal finance tracker for Telegram bot
Commands:
  /mn +500 cà phê      - add income
  /mn -200 ăn trưa     - add expense
  /mn                  - show this month stats
  /mn week             - this week
  /mn all              - all time
  /mn del <id>         - delete entry
"""
import re
import html
from datetime import datetime, timedelta, timezone
from database import add_money_entry, get_money_entries, delete_money_entry

# Simple category detection keywords
CATEGORY_MAP = {
    "ăn": "Ăn uống 🍜",
    "cà phê": "Cà phê ☕",
    "cafe": "Cà phê ☕",
    "coffee": "Cà phê ☕",
    "nhà": "Nhà ở 🏠",
    "điện": "Điện nước 💡",
    "nước": "Điện nước 💡",
    "xăng": "Di chuyển 🚗",
    "grab": "Di chuyển 🚗",
    "taxi": "Di chuyển 🚗",
    "mua": "Mua sắm 🛒",
    "shop": "Mua sắm 🛒",
    "lương": "Thu nhập 💰",
    "thưởng": "Thu nhập 💰",
    "chuyển": "Chuyển khoản 💸",
    "học": "Giáo dục 📚",
    "sách": "Giáo dục 📚",
    "thuốc": "Sức khỏe 💊",
    "khám": "Sức khỏe 💊",
    "game": "Giải trí 🎮",
    "phim": "Giải trí 🎮",
}


def _guess_category(note: str) -> str:
    note_lower = note.lower()
    for keyword, category in CATEGORY_MAP.items():
        if keyword in note_lower:
            return category
    return "Khác 📌"


def _format_amount(amount: float) -> str:
    if amount >= 0:
        return f"+{amount:,.0f}đ"
    return f"{amount:,.0f}đ"


def _format_money_list(entries: list[dict], title: str) -> str:
    if not entries:
        return f"📭 <b>{html.escape(title)}</b>\nChưa có giao dịch nào."

    total_income = sum(e["amount"] for e in entries if e["amount"] > 0)
    total_expense = sum(e["amount"] for e in entries if e["amount"] < 0)
    net = total_income + total_expense

    lines = [
        f"💰 <b>{html.escape(title)}</b>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    for e in entries[:30]:  # show up to 30 entries
        amount_str = _format_amount(e["amount"])
        sign_emoji = "📈" if e["amount"] >= 0 else "📉"
        note = html.escape(e["note"] or "")
        cat = html.escape(e.get("category") or "")
        created = e["created_at"][:10] if e.get("created_at") else ""
        lines.append(f"{sign_emoji} <code>[{e['id']}]</code> <code>{amount_str}</code> — {note} <i>({cat}) {created}</i>")

    lines += [
        "━━━━━━━━━━━━━━━━━━━━",
        f"📈 Thu:  <b>+{total_income:,.0f}đ</b>",
        f"📉 Chi:  <b>{total_expense:,.0f}đ</b>",
        f"💵 Còn:  <b>{_format_amount(net)}</b>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    return "\n".join(lines)


async def handle_money_command(user_id: int, args_text: str) -> str:
    """
    Parse and execute /mn command.
    args_text is everything after '/mn'.
    """
    text = args_text.strip()

    # /mn del <id>
    del_match = re.match(r'^del\s+(\d+)$', text, re.IGNORECASE)
    if del_match:
        entry_id = int(del_match.group(1))
        ok = await delete_money_entry(entry_id, user_id)
        if ok:
            return f"🗑 Đã xóa giao dịch <code>#{entry_id}</code>."
        return f"❌ Không tìm thấy giao dịch <code>#{entry_id}</code> của bạn."

    # /mn week
    if text.lower() == "week":
        since = (datetime.utcnow() - timedelta(days=7)).isoformat()
        entries = await get_money_entries(user_id, since_iso=since)
        return _format_money_list(entries, "Tuần này")

    # /mn all
    if text.lower() == "all":
        entries = await get_money_entries(user_id)
        return _format_money_list(entries, "Tất cả giao dịch")

    # /mn (no args) → this month
    if not text:
        now = datetime.utcnow()
        since = datetime(now.year, now.month, 1).isoformat()
        entries = await get_money_entries(user_id, since_iso=since)
        month_name = now.strftime("%m/%Y")
        return _format_money_list(entries, f"Tháng {month_name}")

    # /mn +500 cà phê  OR  /mn -200 ăn trưa  OR  /mn 500 note
    amount_match = re.match(r'^([+\-]?\d+(?:[.,]\d+)?)\s*(.*)?$', text)
    if amount_match:
        raw_amount = amount_match.group(1).replace(",", ".")
        note = (amount_match.group(2) or "").strip()
        try:
            amount = float(raw_amount)
        except ValueError:
            return "❌ Số tiền không hợp lệ. Ví dụ: <code>/mn +500 cà phê</code>"

        if amount == 0:
            return "❌ Số tiền không được bằng 0."

        category = _guess_category(note)
        entry_id = await add_money_entry(user_id, amount, note, category)
        sign = "📈 Thu" if amount > 0 else "📉 Chi"
        return (
            f"{sign}: <b>{_format_amount(amount)}</b>\n"
            f"Ghi chú: {html.escape(note)}\n"
            f"Danh mục: {html.escape(category)}\n"
            f"ID: <code>#{entry_id}</code>"
        )

    return (
        "❓ Cú pháp không đúng. Xem hướng dẫn:\n\n"
        "<code>/mn +500 cà phê</code> — thêm thu nhập\n"
        "<code>/mn -200 ăn trưa</code> — thêm chi tiêu\n"
        "<code>/mn</code> — thống kê tháng này\n"
        "<code>/mn week</code> — tuần này\n"
        "<code>/mn all</code> — tất cả\n"
        "<code>/mn del &lt;id&gt;</code> — xóa giao dịch"
    )
