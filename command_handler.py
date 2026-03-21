"""
command_handler.py - All slash command handlers for the Telegram bot
"""
import html
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from database import (
    clear_history, get_setting, set_setting,
    get_profile, set_profile, get_history
)
from prompts import MODEL_REGISTRY
from tracker_core import get_usage_report
from money_tracker import handle_money_command
from reminder_system import set_reminder_from_text, list_reminders_text, delete_reminder
from agents_workflow import run_multi_agent_workflow, run_pro_workflow, run_agentic_loop, run_coder_workflow
from rag_core import list_docs, delete_doc

logger = logging.getLogger(__name__)


# ── /start ────────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = update.effective_user.first_name or "bạn"
    await update.message.reply_html(
        f"👋 Xin chào <b>{html.escape(name)}</b>!\n\n"
        f"Tôi là <b>Ultra Bolt</b> — trợ lý AI đa năng 🤖\n\n"
        f"<b>📌 Lệnh cơ bản:</b>\n"
        f"/help — Xem tất cả lệnh\n"
        f"/model — Chọn AI model\n"
        f"/auto — Bật/tắt tự động chọn model\n"
        f"/clear — Xóa lịch sử chat\n"
        f"/profile — Cài đặt hồ sơ cá nhân\n\n"
        f"<b>🔧 Tính năng nâng cao:</b>\n"
        f"/remind — Đặt nhắc nhở\n"
        f"/mn — Quản lý tài chính\n"
        f"/tokens — Thống kê token\n"
        f"/pro — Phân tích sâu\n"
        f"/agent — Agentic AI\n"
        f"/coder — Workflow lập trình\n\n"
        f"💬 Hoặc chỉ cần nhắn tin tự nhiên!"
    )


# ── /help ─────────────────────────────────────────────────────────────────────

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_html(
        "📖 <b>Danh sách lệnh đầy đủ:</b>\n\n"
        "<b>💬 Chat:</b>\n"
        "/clear — Xóa lịch sử hội thoại\n"
        "/model — Chọn model AI\n"
        "/models — Danh sách tất cả models\n"
        "/auto — Bật/tắt tự động chọn model\n"
        "/profile [text] — Xem/đặt hồ sơ cá nhân\n"
        "/stats — Thống kê phiên chat\n\n"
        "<b>🤖 AI Nâng cao:</b>\n"
        "/pro &lt;task&gt; — Phân tích sâu với Mistral Large\n"
        "/agent &lt;task&gt; — Agentic loop tự động (5 bước)\n"
        "/coder &lt;task&gt; — Workflow lập trình chuyên sâu\n\n"
        "<b>⏰ Nhắc nhở:</b>\n"
        "/remind &lt;text&gt; — Đặt nhắc nhở (hỗ trợ tiếng Việt)\n"
        "/reminders — Danh sách nhắc nhở\n\n"
        "<b>💰 Tài chính:</b>\n"
        "/mn +500 cà phê — Thêm thu nhập\n"
        "/mn -200 ăn trưa — Thêm chi tiêu\n"
        "/mn — Thống kê tháng này\n"
        "/mn week — Tuần này\n"
        "/mn all — Tất cả\n"
        "/mn del &lt;id&gt; — Xóa giao dịch\n\n"
        "<b>📚 RAG (Tài liệu):</b>\n"
        "/rag list — Danh sách tài liệu\n"
        "/rag clear &lt;tên file&gt; — Xóa tài liệu\n"
        "Gửi file PDF/TXT/DOCX để thêm vào RAG\n\n"
        "<b>📊 Thống kê:</b>\n"
        "/tokens — Chi phí và token đã dùng\n"
        "/stats — Thống kê phiên hiện tại\n"
    )


# ── /clear ────────────────────────────────────────────────────────────────────

async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Xác nhận xóa", callback_data="clear_yes"),
            InlineKeyboardButton("❌ Hủy", callback_data="clear_no"),
        ]
    ])
    await update.message.reply_html(
        "⚠️ Bạn có chắc muốn xóa toàn bộ lịch sử hội thoại?",
        reply_markup=keyboard
    )


# ── /model ────────────────────────────────────────────────────────────────────

async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    current = await get_setting(user_id, "model_key", "small")

    buttons = []
    for key, info in MODEL_REGISTRY.items():
        checkmark = " ✓" if key == current else ""
        buttons.append([InlineKeyboardButton(
            f"{info['name']}{checkmark} — {info['desc']}",
            callback_data=f"model_{key}"
        )])
    buttons.append([InlineKeyboardButton("❌ Hủy", callback_data="model_cancel")])

    await update.message.reply_html(
        f"🤖 Model hiện tại: <b>{MODEL_REGISTRY.get(current, {}).get('name', current)}</b>\n\nChọn model:",
        reply_markup=InlineKeyboardMarkup(buttons)
    )


# ── /models ───────────────────────────────────────────────────────────────────

async def cmd_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    current = await get_setting(user_id, "model_key", "small")
    lines = ["🤖 <b>Danh sách models:</b>\n"]
    for key, info in MODEL_REGISTRY.items():
        active = " <b>(đang dùng)</b>" if key == current else ""
        lines.append(f"<b>{info['name']}</b>{active}\n  <i>{info['desc']}</i>\n  ID: <code>{info['model_id']}</code>\n")
    await update.message.reply_html("\n".join(lines))


# ── /auto ─────────────────────────────────────────────────────────────────────

async def cmd_auto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    current = await get_setting(user_id, "auto_mode", "1")
    new_val = "0" if current == "1" else "1"
    await set_setting(user_id, "auto_mode", new_val)

    if new_val == "1":
        await update.message.reply_html(
            "🔄 <b>Tự động chọn model: BẬT</b>\n\n"
            "Bot sẽ tự chọn model phù hợp nhất cho mỗi tin nhắn."
        )
    else:
        model_key = await get_setting(user_id, "model_key", "small")
        model_name = MODEL_REGISTRY.get(model_key, {}).get("name", model_key)
        await update.message.reply_html(
            f"🔒 <b>Tự động chọn model: TẮT</b>\n\n"
            f"Sẽ dùng cố định: <b>{model_name}</b>\n"
            f"Dùng /model để thay đổi."
        )


# ── /profile ──────────────────────────────────────────────────────────────────

async def cmd_profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args = " ".join(context.args) if context.args else ""

    if args:
        await set_profile(user_id, args)
        await update.message.reply_html(
            f"✅ Đã cập nhật hồ sơ:\n<i>{html.escape(args)}</i>"
        )
    else:
        profile = await get_profile(user_id)
        if profile:
            await update.message.reply_html(
                f"👤 <b>Hồ sơ của bạn:</b>\n<i>{html.escape(profile)}</i>\n\n"
                f"Dùng <code>/profile [nội dung]</code> để cập nhật."
            )
        else:
            await update.message.reply_html(
                "👤 Bạn chưa có hồ sơ.\n\n"
                "Dùng <code>/profile [mô tả về bạn]</code> để thêm.\n"
                "Ví dụ: <code>/profile Tôi là lập trình viên Python, thích toán học</code>"
            )


# ── /stats ────────────────────────────────────────────────────────────────────

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    history = await get_history(user_id, limit=100)
    model_key = await get_setting(user_id, "model_key", "small")
    auto_mode = await get_setting(user_id, "auto_mode", "1")

    user_msgs = sum(1 for m in history if m["role"] == "user")
    bot_msgs = sum(1 for m in history if m["role"] == "assistant")
    model_name = MODEL_REGISTRY.get(model_key, {}).get("name", model_key)
    auto_str = "BẬT" if auto_mode == "1" else "TẮT"

    await update.message.reply_html(
        f"📊 <b>Thống kê của bạn:</b>\n\n"
        f"💬 Tổng tin nhắn: <b>{len(history)}</b>\n"
        f"👤 Câu hỏi: <b>{user_msgs}</b>\n"
        f"🤖 Phản hồi: <b>{bot_msgs}</b>\n"
        f"🧠 Model: <b>{model_name}</b>\n"
        f"🔄 Auto-routing: <b>{auto_str}</b>"
    )


# ── /remind ───────────────────────────────────────────────────────────────────

async def cmd_remind(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = " ".join(context.args) if context.args else ""

    if not text:
        await update.message.reply_html(
            "⏰ <b>Đặt nhắc nhở:</b>\n\n"
            "Cú pháp: <code>/remind [nội dung] [thời gian]</code>\n\n"
            "Ví dụ:\n"
            "<code>/remind họp team lúc 9h sáng ngày mai</code>\n"
            "<code>/remind uống thuốc 7h tối</code>\n"
            "<code>/remind kiểm tra email mỗi ngày 8h sáng</code>\n"
            "<code>/remind gửi báo cáo sau 30 phút</code>"
        )
        return

    await update.message.reply_html("⏳ Đang xử lý nhắc nhở...")
    result = await set_reminder_from_text(user_id, text)
    await update.message.reply_html(result)


# ── /reminders ────────────────────────────────────────────────────────────────

async def cmd_reminders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = await list_reminders_text(user_id)
    await update.message.reply_html(text)


# ── /mn ───────────────────────────────────────────────────────────────────────

async def cmd_mn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args_text = " ".join(context.args) if context.args else ""
    result = await handle_money_command(user_id, args_text)
    await update.message.reply_html(result)


# ── /pro ──────────────────────────────────────────────────────────────────────

async def cmd_pro(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    task = " ".join(context.args) if context.args else ""

    if not task:
        await update.message.reply_html(
            "🔬 <b>Pro Workflow</b>\n\n"
            "Dùng Mistral Large để phân tích sâu và tổng hợp.\n\n"
            "Cú pháp: <code>/pro [câu hỏi/task]</code>\n"
            "Ví dụ: <code>/pro So sánh React vs Vue vs Angular 2024</code>"
        )
        return

    await update.message.reply_html("🔬 <b>Pro Workflow đang chạy...</b>")
    result = await run_pro_workflow(user_id, task)
    # Split if too long
    if len(result) > 4000:
        parts = [result[i:i+4000] for i in range(0, len(result), 4000)]
        for part in parts:
            await update.message.reply_html(part)
    else:
        await update.message.reply_html(result)


# ── /agent ────────────────────────────────────────────────────────────────────

async def cmd_agent(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    task = " ".join(context.args) if context.args else ""

    if not task:
        await update.message.reply_html(
            "🤖 <b>Agentic Loop</b>\n\n"
            "AI tự động thực hiện nhiệm vụ qua nhiều bước (tối đa 5 vòng lặp).\n\n"
            "Cú pháp: <code>/agent [nhiệm vụ]</code>\n"
            "Ví dụ: <code>/agent Lên kế hoạch học Python trong 30 ngày</code>"
        )
        return

    await update.message.reply_html("🤖 <b>Agentic Loop đang chạy...</b> (có thể mất 30-60 giây)")
    result = await run_agentic_loop(user_id, task)
    if len(result) > 4000:
        parts = [result[i:i+4000] for i in range(0, len(result), 4000)]
        for part in parts:
            await update.message.reply_html(part)
    else:
        await update.message.reply_html(result)


# ── /coder ────────────────────────────────────────────────────────────────────

async def cmd_coder(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    task = " ".join(context.args) if context.args else ""

    if not task:
        await update.message.reply_html(
            "💻 <b>Coder Workflow</b>\n\n"
            "Thiết kế + Implement + Review code tự động.\n\n"
            "Cú pháp: <code>/coder [yêu cầu]</code>\n"
            "Ví dụ: <code>/coder Viết REST API FastAPI CRUD cho user</code>"
        )
        return

    await update.message.reply_html("💻 <b>Coder Workflow đang chạy...</b> (có thể mất 30-60 giây)")
    result = await run_coder_workflow(user_id, task)
    if len(result) > 4000:
        parts = [result[i:i+4000] for i in range(0, len(result), 4000)]
        for part in parts:
            await update.message.reply_html(part)
    else:
        await update.message.reply_html(result)


# ── /rag ──────────────────────────────────────────────────────────────────────

async def cmd_rag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args = context.args or []

    if not args or args[0].lower() == "list":
        text = await list_docs(user_id)
        await update.message.reply_html(text)

    elif args[0].lower() == "clear" and len(args) >= 2:
        filename = " ".join(args[1:])
        result = await delete_doc(user_id, filename)
        await update.message.reply_html(result)

    else:
        await update.message.reply_html(
            "📚 <b>RAG Commands:</b>\n\n"
            "<code>/rag list</code> — Danh sách tài liệu\n"
            "<code>/rag clear &lt;tên file&gt;</code> — Xóa tài liệu\n\n"
            "Gửi file PDF/TXT/DOCX để thêm vào knowledge base."
        )


# ── /tokens ───────────────────────────────────────────────────────────────────

async def cmd_tokens(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    report = await get_usage_report(user_id)
    await update.message.reply_html(report)


# ── Callback query handler ────────────────────────────────────────────────────

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    data = query.data

    # Clear confirmation
    if data == "clear_yes":
        await clear_history(user_id)
        await query.edit_message_text(
            "🗑 <b>Đã xóa lịch sử hội thoại.</b>",
            parse_mode=ParseMode.HTML
        )
    elif data == "clear_no":
        await query.edit_message_text("↩️ Đã hủy.")

    # Model selection
    elif data.startswith("model_"):
        key = data[len("model_"):]
        if key == "cancel":
            await query.edit_message_text("↩️ Đã hủy.")
        elif key in MODEL_REGISTRY:
            await set_setting(user_id, "model_key", key)
            info = MODEL_REGISTRY[key]
            await query.edit_message_text(
                f"✅ Đã chuyển sang <b>{info['name']}</b>\n"
                f"<i>{info['desc']}</i>",
                parse_mode=ParseMode.HTML
            )
        else:
            await query.edit_message_text("❌ Model không hợp lệ.")

    # Reminder delete confirmation
    elif data.startswith("remind_del_"):
        reminder_id = int(data[len("remind_del_"):])
        await delete_reminder(reminder_id)
        await query.edit_message_text(
            f"🗑 Đã xóa nhắc nhở <code>#{reminder_id}</code>.",
            parse_mode=ParseMode.HTML
        )
