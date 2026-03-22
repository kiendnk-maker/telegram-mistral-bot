"""
command_handler.py - All slash command handlers for the Telegram bot
"""
import html
import logging
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode, ChatAction
from telegram.ext import ContextTypes

from database import (
    clear_history, get_setting, set_setting,
    get_profile, set_profile, get_history,
    add_allowed_user, remove_allowed_user, list_allowed_users,
)
from prompts import MODEL_REGISTRY
from tracker_core import get_usage_report
from money_tracker import handle_money_command
from reminder_system import set_reminder_from_text, list_reminders_text, delete_reminder
from agents_workflow import run_multi_agent_workflow, run_pro_workflow, run_agentic_loop, run_coder_workflow
from rag_core import list_docs, delete_doc
from focus_tracker import (
    add_task, get_tasks, complete_task, delete_task, clear_done_tasks,
    build_task_list, get_daily_summary, get_motivation,
    start_pomodoro, get_pomodoro_count_today,
)

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency — bot.py defines MAIN_MENU
def _main_menu():
    from bot import MAIN_MENU
    return MAIN_MENU


# ── /start ────────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = update.effective_user.first_name or "bạn"

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🤖 Chọn Model",   callback_data="settings_model"),
            InlineKeyboardButton("📖 Xem lệnh",      callback_data="start_help"),
        ],
        [
            InlineKeyboardButton("💬 Bắt đầu chat",  callback_data="start_chat"),
        ],
    ])

    await update.message.reply_html(
        f"⚡ <b>Ultra Bolt</b> đã sẵn sàng!\n\n"
        f"Xin chào, <b>{html.escape(name)}</b>! Tôi là trợ lý AI powered by <b>Mistral AI</b>.\n\n"
        f"🧠 <b>Tính năng:</b>\n"
        f"• Chat thông minh với auto-routing model\n"
        f"• Xử lý ảnh &amp; giọng nói\n"
        f"• Đặt nhắc nhở bằng ngôn ngữ tự nhiên\n"
        f"• Theo dõi chi tiêu cá nhân\n"
        f"• Upload tài liệu &amp; hỏi nội dung (RAG)\n"
        f"• Multi-agent AI workflows",
        reply_markup=_main_menu(),
    )
    # Send inline action buttons as a separate message so the keyboard stays clean
    await update.message.reply_html(
        "Chọn một tùy chọn để bắt đầu:",
        reply_markup=keyboard,
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
        "/stats — Thống kê phiên hiện tại\n",
        reply_markup=_main_menu(),
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
    current = await get_setting(user_id, "model_key", "groq_large")

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
    current = await get_setting(user_id, "model_key", "groq_large")
    text = "🤖 <b>Danh sách models:</b>\n\n"
    for key, info in MODEL_REGISTRY.items():
        active = " ✓" if key == current else ""
        text += f"{info['name']}{active}\n"
        text += f"<i>{info['desc']}</i>\n"
        text += f"<code>{info['model_id']}</code>\n\n"
    await update.message.reply_html(text.strip())


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
        model_key = await get_setting(user_id, "model_key", "groq_large")
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
    model_key = await get_setting(user_id, "model_key", "groq_large")
    auto_mode = await get_setting(user_id, "auto_mode", "1")

    user_msgs = sum(1 for m in history if m["role"] == "user")
    model_name = MODEL_REGISTRY.get(model_key, {}).get("name", model_key)
    auto_str = "BẬT" if auto_mode == "1" else "TẮT"

    await update.message.reply_html(
        "📊 <b>Thống kê của bạn</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"💬 Tin nhắn:  <b>{user_msgs}</b>\n"
        f"🤖 Model:     <b>{html.escape(model_name)}</b>\n"
        f"🔄 Auto:      <b>{auto_str}</b>\n"
        "━━━━━━━━━━━━━━━━━━━━"
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

    progress_msg = await update.message.reply_html(
        "⏳ <b>Pro Workflow đang chạy...</b>\n<i>Có thể mất 30-60 giây</i>"
    )
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    result = await run_pro_workflow(user_id, task)

    # Edit progress message with first chunk, send rest separately
    if len(result) > 4000:
        parts = [result[i:i+4000] for i in range(0, len(result), 4000)]
        await progress_msg.edit_text(parts[0], parse_mode=ParseMode.HTML)
        for part in parts[1:]:
            await update.message.reply_html(part)
    else:
        await progress_msg.edit_text(result, parse_mode=ParseMode.HTML)


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

    progress_msg = await update.message.reply_html(
        "⏳ <b>Agentic Loop đang chạy...</b>\n<i>Có thể mất 30-60 giây</i>"
    )
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    result = await run_agentic_loop(user_id, task)

    if len(result) > 4000:
        parts = [result[i:i+4000] for i in range(0, len(result), 4000)]
        await progress_msg.edit_text(parts[0], parse_mode=ParseMode.HTML)
        for part in parts[1:]:
            await update.message.reply_html(part)
    else:
        await progress_msg.edit_text(result, parse_mode=ParseMode.HTML)


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

    progress_msg = await update.message.reply_html(
        "⏳ <b>Coder Workflow đang chạy...</b>\n<i>Có thể mất 30-60 giây</i>"
    )
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    result = await run_coder_workflow(user_id, task)

    if len(result) > 4000:
        parts = [result[i:i+4000] for i in range(0, len(result), 4000)]
        await progress_msg.edit_text(parts[0], parse_mode=ParseMode.HTML)
        for part in parts[1:]:
            await update.message.reply_html(part)
    else:
        await progress_msg.edit_text(result, parse_mode=ParseMode.HTML)


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

    # ── /start inline buttons ────────────────────────────────────────────────
    if data == "start_help":
        await query.edit_message_text(
            "📖 Dùng menu bên dưới hoặc gõ /help để xem tất cả lệnh.",
            parse_mode=ParseMode.HTML,
        )
        return

    if data == "start_chat":
        await query.edit_message_text(
            "💬 Sẵn sàng! Hãy nhắn tin bất kỳ để bắt đầu.",
            parse_mode=ParseMode.HTML,
        )
        return

    # ── Settings panel callbacks ─────────────────────────────────────────────
    if data == "settings_model":
        current = await get_setting(user_id, "model_key", "groq_large")
        buttons = []
        for key, info in MODEL_REGISTRY.items():
            checkmark = " ✓" if key == current else ""
            buttons.append([InlineKeyboardButton(
                f"{info['name']}{checkmark} — {info['desc']}",
                callback_data=f"model_{key}"
            )])
        buttons.append([InlineKeyboardButton("❌ Hủy", callback_data="model_cancel")])
        await query.edit_message_text(
            f"🤖 Model hiện tại: <b>{MODEL_REGISTRY.get(current, {}).get('name', current)}</b>\n\nChọn model:",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(buttons),
        )
        return

    if data == "settings_auto":
        current = await get_setting(user_id, "auto_mode", "1")
        new_val = "0" if current == "1" else "1"
        await set_setting(user_id, "auto_mode", new_val)
        auto_str = "BẬT" if new_val == "1" else "TẮT"
        await query.edit_message_text(
            f"🔄 Auto-routing đã chuyển sang: <b>{auto_str}</b>",
            parse_mode=ParseMode.HTML,
        )
        return

    if data == "settings_profile":
        profile = await get_profile(user_id)
        if profile:
            await query.edit_message_text(
                f"👤 <b>Hồ sơ hiện tại:</b>\n<i>{html.escape(profile)}</i>\n\n"
                f"Dùng <code>/profile [nội dung mới]</code> để cập nhật.",
                parse_mode=ParseMode.HTML,
            )
        else:
            await query.edit_message_text(
                "👤 Bạn chưa có hồ sơ.\n\n"
                "Dùng <code>/profile [mô tả về bạn]</code> để thêm.\n"
                "Ví dụ: <code>/profile Tôi là lập trình viên Python</code>",
                parse_mode=ParseMode.HTML,
            )
        return

    if data == "settings_clear":
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Xác nhận xóa", callback_data="clear_yes"),
                InlineKeyboardButton("❌ Hủy", callback_data="clear_no"),
            ]
        ])
        await query.edit_message_text(
            "⚠️ Bạn có chắc muốn xóa toàn bộ lịch sử hội thoại?",
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard,
        )
        return

    # ── Clear confirmation ───────────────────────────────────────────────────
    if data == "clear_yes":
        await clear_history(user_id)
        await query.edit_message_text(
            "🗑 <b>Đã xóa lịch sử hội thoại.</b>",
            parse_mode=ParseMode.HTML
        )
    elif data == "clear_no":
        await query.edit_message_text("↩️ Đã hủy.")

    # ── Model selection ──────────────────────────────────────────────────────
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

    # ── Reminder delete confirmation ─────────────────────────────────────────
    elif data.startswith("remind_del_"):
        reminder_id = int(data[len("remind_del_"):])
        await delete_reminder(reminder_id)
        await query.edit_message_text(
            f"🗑 Đã xóa nhắc nhở <code>#{reminder_id}</code>.",
            parse_mode=ParseMode.HTML
        )

    # ── Task done callback ────────────────────────────────────────────────────
    elif data.startswith("task_done_"):
        task_id = int(data[len("task_done_"):])
        ok = await complete_task(user_id, task_id)
        if ok:
            await query.edit_message_text(f"✅ Hoàn thành task <code>#{task_id}</code>! Tốt lắm! 🎉", parse_mode=ParseMode.HTML)
        else:
            await query.edit_message_text("❌ Không tìm thấy task hoặc đã hoàn thành rồi.", parse_mode=ParseMode.HTML)

    # ── Task delete callback ──────────────────────────────────────────────────
    elif data.startswith("task_del_"):
        task_id = int(data[len("task_del_"):])
        ok = await delete_task(user_id, task_id)
        if ok:
            await query.edit_message_text(f"🗑 Đã xóa task <code>#{task_id}</code>.", parse_mode=ParseMode.HTML)
        else:
            await query.edit_message_text("❌ Không tìm thấy task.", parse_mode=ParseMode.HTML)


# ── /todo ─────────────────────────────────────────────────────────────────────

async def cmd_todo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    title = " ".join(context.args) if context.args else ""

    if not title:
        await update.message.reply_html(
            "📝 <b>Thêm việc cần làm:</b>\n\n"
            "Cú pháp: <code>/todo [tên công việc]</code>\n"
            "Ví dụ: <code>/todo Hoàn thành báo cáo tháng 3</code>"
        )
        return

    task_id = await add_task(user_id, title)
    await update.message.reply_html(
        f"✅ Đã thêm task <code>#{task_id}</code>:\n<b>{html.escape(title)}</b>\n\n"
        f"Dùng /tasks để xem danh sách."
    )


# ── /tasks ────────────────────────────────────────────────────────────────────

async def cmd_tasks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    tasks = await get_tasks(user_id, status="pending")

    if not tasks:
        await update.message.reply_html(
            "📋 <b>Danh sách việc cần làm</b>\n\nKhông có việc gì cả! 🎉\nDùng /todo để thêm."
        )
        return

    text = f"📋 <b>Việc cần làm ({len(tasks)}):</b>\n\n"
    buttons = []
    for t in tasks:
        text += f"<code>[{t['id']}]</code> {html.escape(t['title'])}\n"
        buttons.append([
            InlineKeyboardButton(f"✅ #{t['id']} Xong", callback_data=f"task_done_{t['id']}"),
            InlineKeyboardButton(f"🗑 Xóa", callback_data=f"task_del_{t['id']}"),
        ])

    await update.message.reply_html(text, reply_markup=InlineKeyboardMarkup(buttons))


# ── /done ─────────────────────────────────────────────────────────────────────

async def cmd_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_html(
            "Cú pháp: <code>/done [số id]</code>\n"
            "Ví dụ: <code>/done 3</code>\n\n"
            "Dùng /tasks để xem id của task."
        )
        return

    task_id = int(context.args[0])
    ok = await complete_task(user_id, task_id)
    if ok:
        await update.message.reply_html(f"🎉 Hoàn thành task <code>#{task_id}</code>! Tuyệt vời!")
    else:
        await update.message.reply_html("❌ Không tìm thấy task hoặc đã hoàn thành rồi.")


# ── /deltask ──────────────────────────────────────────────────────────────────

async def cmd_deltask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_html("Cú pháp: <code>/deltask [số id]</code>")
        return

    task_id = int(context.args[0])
    ok = await delete_task(user_id, task_id)
    if ok:
        await update.message.reply_html(f"🗑 Đã xóa task <code>#{task_id}</code>.")
    else:
        await update.message.reply_html("❌ Không tìm thấy task.")


# ── /pomodoro ─────────────────────────────────────────────────────────────────

async def cmd_pomodoro(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    task_title = " ".join(context.args) if context.args else ""

    await start_pomodoro(user_id, task_title)
    count_today = await get_pomodoro_count_today(user_id)

    task_text = f"\n🎯 Task: <b>{html.escape(task_title)}</b>" if task_title else ""
    await update.message.reply_html(
        f"🍅 <b>Pomodoro bắt đầu!</b>{task_text}\n\n"
        f"⏱ Tập trung làm việc trong <b>25 phút</b>.\n"
        f"📵 Tắt thông báo, đừng để bị xao nhãng.\n\n"
        f"Hôm nay bạn đã làm: <b>{count_today}</b> pomodoro\n\n"
        f"<i>Bot sẽ nhắc bạn sau 25 phút...</i>"
    )

    # Đặt reminder 25 phút
    from reminder_system import set_reminder_from_text
    await set_reminder_from_text(
        user_id,
        f"🍅 Pomodoro xong! Nghỉ 5 phút nhé{' - ' + task_title if task_title else ''}. sau 25 phút"
    )


# ── /motivation ───────────────────────────────────────────────────────────────

async def cmd_motivation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_html(get_motivation())


# ── /checkin ──────────────────────────────────────────────────────────────────

async def cmd_checkin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    summary = await get_daily_summary(user_id)
    await update.message.reply_html(summary)


# ── /user ─────────────────────────────────────────────────────────────────────

async def cmd_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /user add <id>    — thêm user vào whitelist
    /user remove <id> — xóa user khỏi whitelist
    /user list        — xem danh sách
    Chỉ owner mới dùng được lệnh này.
    """
    import os
    owner_id = int(os.getenv("OWNER_ID", "0"))
    caller_id = update.effective_user.id

    if owner_id and caller_id != owner_id:
        await update.message.reply_html("⛔ Chỉ owner mới dùng được lệnh này.")
        return

    args = context.args or []
    sub = args[0].lower() if args else ""

    if sub == "add" and len(args) >= 2:
        if not args[1].lstrip("-").isdigit():
            await update.message.reply_html("❌ ID phải là số nguyên.")
            return
        target_id = int(args[1])
        added = await add_allowed_user(target_id, caller_id)
        if added:
            await update.message.reply_html(f"✅ Đã thêm <code>{target_id}</code> vào danh sách cho phép.")
        else:
            await update.message.reply_html(f"ℹ️ <code>{target_id}</code> đã có trong danh sách rồi.")

    elif sub == "remove" and len(args) >= 2:
        if not args[1].lstrip("-").isdigit():
            await update.message.reply_html("❌ ID phải là số nguyên.")
            return
        target_id = int(args[1])
        removed = await remove_allowed_user(target_id)
        if removed:
            await update.message.reply_html(f"🗑 Đã xóa <code>{target_id}</code> khỏi danh sách.")
        else:
            await update.message.reply_html(f"❌ Không tìm thấy <code>{target_id}</code>.")

    elif sub == "list":
        users = await list_allowed_users()
        if not users:
            await update.message.reply_html("📭 Danh sách trống. Chỉ owner đang có quyền truy cập.")
            return
        lines = [f"👥 <b>Người dùng được phép ({len(users)}):</b>\n"]
        for u in users:
            lines.append(f"• <code>{u['user_id']}</code> — thêm lúc {u['added_at'][:16]}")
        await update.message.reply_html("\n".join(lines))

    else:
        await update.message.reply_html(
            "👥 <b>Quản lý người dùng</b>\n\n"
            "<code>/user add &lt;id&gt;</code> — Thêm người dùng\n"
            "<code>/user remove &lt;id&gt;</code> — Xóa người dùng\n"
            "<code>/user list</code> — Xem danh sách\n\n"
            "<i>Chỉ owner mới dùng được lệnh này.</i>"
        )
