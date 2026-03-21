"""
bot.py - Main entry point for Ultra Bolt Telegram bot
"""
import os
import re
import html
import time
import logging
import asyncio
import base64
import tempfile
from collections import defaultdict

from dotenv import load_dotenv
from telegram import Update, BotCommand, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)
from telegram.constants import ChatAction, ParseMode
from telegram.error import TelegramError

from database import init_db
from money_tracker import handle_money_command
from llm_core import call_llm, call_vision, transcribe_audio
from rag_core import has_docs, add_document, build_rag_context
from reminder_system import reminder_loop
from command_handler import (
    cmd_start, cmd_help, cmd_clear, cmd_model, cmd_models,
    cmd_auto, cmd_profile, cmd_stats, cmd_remind, cmd_reminders,
    cmd_mn, cmd_pro, cmd_agent, cmd_coder, cmd_rag, cmd_tokens,
    handle_callback,
)
from prompts import MODEL_REGISTRY

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("Thiếu TELEGRAM_TOKEN trong file .env")
if not os.getenv("MISTRAL_API_KEY"):
    raise RuntimeError("Thiếu MISTRAL_API_KEY trong file .env")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Rate limiting: max 10 messages per 60 seconds per user
RATE_LIMIT = 10
RATE_WINDOW = 60
MAX_MSG_LENGTH = 4000

_rate_tracker: dict[int, list[float]] = defaultdict(list)

# ── Persistent Reply Keyboard ─────────────────────────────────────────────────

MAIN_MENU = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton("💬 Chat mới"),    KeyboardButton("🤖 Đổi Model")],
        [KeyboardButton("💰 Chi tiêu"),    KeyboardButton("⏰ Nhắc nhở")],
        [KeyboardButton("📚 Tài liệu"),    KeyboardButton("📊 Thống kê")],
        [KeyboardButton("⚙️ Cài đặt"),    KeyboardButton("❓ Trợ giúp")],
    ],
    resize_keyboard=True,
    persistent=True,
)

# Set of menu button texts for fast lookup
MENU_BUTTONS = {
    "💬 Chat mới",
    "🤖 Đổi Model",
    "💰 Chi tiêu",
    "⏰ Nhắc nhở",
    "📚 Tài liệu",
    "📊 Thống kê",
    "⚙️ Cài đặt",
    "❓ Trợ giúp",
}


def _is_rate_limited(user_id: int) -> tuple[bool, int]:
    now = time.time()
    timestamps = _rate_tracker[user_id]
    _rate_tracker[user_id] = [t for t in timestamps if now - t < RATE_WINDOW]
    if len(_rate_tracker[user_id]) >= RATE_LIMIT:
        wait = int(RATE_WINDOW - (now - _rate_tracker[user_id][0])) + 1
        return True, wait
    _rate_tracker[user_id].append(now)
    return False, 0


def _md_to_html(text: str) -> str:
    """Convert common markdown to Telegram HTML safely."""
    code_blocks: list[str] = []
    inline_codes: list[str] = []

    def save_code_block(m):
        code_blocks.append(html.escape(m.group(1)))
        return f"%%CB{len(code_blocks) - 1}%%"

    def save_inline_code(m):
        inline_codes.append(html.escape(m.group(1)))
        return f"%%IC{len(inline_codes) - 1}%%"

    # Save code blocks first (before escaping)
    text = re.sub(r'```[\w]*\n?([\s\S]+?)```', save_code_block, text)
    text = re.sub(r'`([^`\n]+)`', save_inline_code, text)

    # Escape HTML special chars
    text = html.escape(text)

    # Convert markdown headings (#### ## #) → bold
    text = re.sub(r'^#{1,6}\s+(.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)

    # Bold: **text** (including multiline)
    text = re.sub(r'\*\*([\s\S]+?)\*\*', r'<b>\1</b>', text)

    # Italic: *text* (single line only to avoid false positives)
    text = re.sub(r'\*([^*\n]+?)\*', r'<i>\1</i>', text)

    # Underline: __text__
    text = re.sub(r'__([\s\S]+?)__', r'<u>\1</u>', text)

    # Strip leftover markdown list markers that didn't convert
    text = re.sub(r'^[ \t]*[-•]\s+', '• ', text, flags=re.MULTILINE)

    # Restore code
    for i, code in enumerate(inline_codes):
        text = text.replace(f"%%IC{i}%%", f"<code>{code}</code>")
    for i, code in enumerate(code_blocks):
        text = text.replace(f"%%CB{i}%%", f"<pre>{code}</pre>")

    return text


async def _send_long(update: Update, text: str, parse_mode: str = ParseMode.HTML, reply_markup=None):
    """Send message, splitting at 4000 chars if needed."""
    if len(text) <= 4000:
        await update.message.reply_text(text, parse_mode=parse_mode, reply_markup=reply_markup)
        return
    parts = [text[i:i+4000] for i in range(0, len(text), 4000)]
    for idx, part in enumerate(parts):
        # Only attach reply_markup to the last chunk
        markup = reply_markup if idx == len(parts) - 1 else None
        await update.message.reply_text(part, parse_mode=parse_mode, reply_markup=markup)


# ── Text message handler ──────────────────────────────────────────────────────

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_message = update.message.text

    # ── Menu button routing ──────────────────────────────────────────────────
    if user_message in MENU_BUTTONS:
        if user_message == "💬 Chat mới":
            from database import clear_history as _clear_history
            await _clear_history(user_id)
            await update.message.reply_html(
                "🗑 Đã xóa lịch sử. Bắt đầu cuộc hội thoại mới!",
                reply_markup=MAIN_MENU,
            )
        elif user_message == "🤖 Đổi Model":
            await cmd_model(update, context)
        elif user_message == "💰 Chi tiêu":
            result = await handle_money_command(update.effective_user.id, "")
            await update.message.reply_html(result)
        elif user_message == "⏰ Nhắc nhở":
            await cmd_reminders(update, context)
        elif user_message == "📚 Tài liệu":
            await cmd_rag(update, context)
        elif user_message == "📊 Thống kê":
            await cmd_stats(update, context)
        elif user_message == "⚙️ Cài đặt":
            await _show_settings(update, context)
        elif user_message == "❓ Trợ giúp":
            await cmd_help(update, context)
        return

    # ── Normal LLM flow ──────────────────────────────────────────────────────
    if len(user_message) > MAX_MSG_LENGTH:
        await update.message.reply_html(
            f"⚠️ Tin nhắn quá dài (<b>{len(user_message)}/{MAX_MSG_LENGTH}</b> ký tự)."
        )
        return

    limited, wait_sec = _is_rate_limited(user_id)
    if limited:
        await update.message.reply_html(
            f"⏳ Bạn gửi quá nhanh! Vui lòng chờ <b>{wait_sec} giây</b>."
        )
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    start_time = time.time()

    try:
        # Check for RAG context
        extra_context = None
        if await has_docs(user_id):
            extra_context = await build_rag_context(user_id, user_message)

        reply, model_key = await call_llm(user_id, user_message, extra_context=extra_context)
        elapsed = round(time.time() - start_time, 1)

        reply_html = _md_to_html(reply)
        model_name = MODEL_REGISTRY.get(model_key, {}).get("name", model_key)
        rag_badge = " 📚" if extra_context else ""
        footer = f"\n\n<i>⏱ {elapsed}s · {model_name}{rag_badge}</i>"

        await _send_long(update, reply_html + footer)
        logger.info(f"User {user_id} | {elapsed}s | model={model_key} | {len(user_message)} chars")

    except asyncio.TimeoutError:
        await update.message.reply_html(
            "⏰ <b>Yêu cầu mất quá nhiều thời gian.</b>\n"
            "Thử câu ngắn hơn hoặc dùng /model để chọn Mistral Small."
        )
    except Exception as e:
        logger.error(f"Chat error user {user_id}: {e}", exc_info=True)
        await update.message.reply_html(
            "⚠️ <b>Có lỗi xảy ra.</b> Vui lòng thử lại.\n"
            "Nếu lỗi tiếp tục, dùng /clear để reset hội thoại."
        )


# ── Settings panel ────────────────────────────────────────────────────────────

async def _show_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from database import get_setting, get_profile
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    user_id = update.effective_user.id
    model_key = await get_setting(user_id, "model_key", "small")
    auto_mode = await get_setting(user_id, "auto_mode", "1")
    profile = await get_profile(user_id)

    model_name = MODEL_REGISTRY.get(model_key, {}).get("name", model_key)
    auto_str = "BẬT" if auto_mode == "1" else "TẮT"
    profile_str = "Đã cài" if profile else "Chưa cài"

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🤖 Đổi Model",    callback_data="settings_model"),
            InlineKeyboardButton("🔄 Toggle Auto",   callback_data="settings_auto"),
        ],
        [
            InlineKeyboardButton("👤 Sửa hồ sơ",    callback_data="settings_profile"),
            InlineKeyboardButton("🗑 Xóa lịch sử",  callback_data="settings_clear"),
        ],
    ])

    await update.message.reply_html(
        "⚙️ <b>Cài đặt</b>\n\n"
        f"🤖 Model: <b>{html.escape(model_name)}</b>\n"
        f"🔄 Auto-routing: <b>{auto_str}</b>\n"
        f"👤 Hồ sơ: <b>{profile_str}</b>",
        reply_markup=keyboard,
    )


# ── Voice message handler ─────────────────────────────────────────────────────

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    limited, wait_sec = _is_rate_limited(user_id)
    if limited:
        await update.message.reply_html(f"⏳ Chờ <b>{wait_sec} giây</b> trước khi gửi tiếp.")
        return

    if not os.getenv("GROQ_API_KEY"):
        await update.message.reply_html(
            "⚠️ Tính năng giọng nói chưa được cấu hình (thiếu GROQ_API_KEY)."
        )
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    tmp_path = None
    try:
        voice = update.message.voice
        voice_file = await context.bot.get_file(voice.file_id)

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp_path = tmp.name
        await voice_file.download_to_drive(tmp_path)

        transcript = await transcribe_audio(tmp_path)
        if not transcript or not transcript.strip():
            await update.message.reply_html(
                "🎙 Không nhận diện được giọng nói. Vui lòng thử lại."
            )
            return

        await update.message.reply_html(f"🎙 <i>Đã nghe: {html.escape(transcript)}</i>")

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

        extra_context = None
        if await has_docs(user_id):
            extra_context = await build_rag_context(user_id, transcript)

        reply, model_key = await call_llm(user_id, transcript, extra_context=extra_context)
        reply_html = _md_to_html(reply)
        model_name = MODEL_REGISTRY.get(model_key, {}).get("name", model_key)
        footer = f"\n\n<i>🎙 STT · {model_name}</i>"
        await _send_long(update, reply_html + footer)

    except Exception as e:
        logger.error(f"Voice error user {user_id}: {e}", exc_info=True)
        await update.message.reply_html("⚠️ Lỗi xử lý giọng nói. Vui lòng thử lại.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Photo message handler ─────────────────────────────────────────────────────

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    limited, wait_sec = _is_rate_limited(user_id)
    if limited:
        await update.message.reply_html(f"⏳ Chờ <b>{wait_sec} giây</b> trước khi gửi tiếp.")
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    tmp_path = None
    try:
        photo = update.message.photo[-1]  # largest size
        photo_file = await context.bot.get_file(photo.file_id)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        await photo_file.download_to_drive(tmp_path)

        with open(tmp_path, "rb") as f:
            image_bytes = f.read()

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt = update.message.caption or "Mô tả chi tiết hình ảnh này bằng tiếng Việt."

        reply = await call_vision(user_id, image_b64, prompt)
        reply_html = _md_to_html(reply)
        footer = "\n\n<i>👁 Pixtral Vision</i>"
        await _send_long(update, reply_html + footer)

    except Exception as e:
        logger.error(f"Photo error user {user_id}: {e}", exc_info=True)
        await update.message.reply_html("⚠️ Lỗi xử lý ảnh. Vui lòng thử lại.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Document handler (for RAG) ────────────────────────────────────────────────

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    doc = update.message.document

    if not doc:
        return

    filename = doc.file_name or "unknown"
    ext = os.path.splitext(filename)[1].lower()
    allowed_exts = {".pdf", ".txt", ".docx", ".doc", ".md", ".csv"}

    if ext not in allowed_exts:
        await update.message.reply_html(
            f"❌ Định dạng không hỗ trợ: <code>{html.escape(ext)}</code>\n"
            f"Cho phép: PDF, TXT, DOCX, MD, CSV"
        )
        return

    if doc.file_size and doc.file_size > 10 * 1024 * 1024:
        size_mb = doc.file_size / (1024 * 1024)
        await update.message.reply_html(f"❌ File quá lớn ({size_mb:.1f}MB). Giới hạn 10MB.")
        return

    await update.message.reply_html(f"📄 Đang xử lý <b>{html.escape(filename)}</b>...")

    tmp_path = None
    try:
        file = await context.bot.get_file(doc.file_id)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
        await file.download_to_drive(tmp_path)

        with open(tmp_path, "rb") as f:
            content = f.read()

        result = await add_document(user_id, filename, content)
        await update.message.reply_html(result)

    except Exception as e:
        logger.error(f"Document error user {user_id}: {e}", exc_info=True)
        await update.message.reply_html("⚠️ Lỗi xử lý tài liệu. Vui lòng thử lại.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Error handler ─────────────────────────────────────────────────────────────

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Unhandled error: {context.error}", exc_info=context.error)


# ── Post-init: DB + reminder loop ─────────────────────────────────────────────

async def post_init(application):
    await init_db()
    asyncio.create_task(reminder_loop(application.bot))

    # Register commands in Telegram "/" menu
    await application.bot.set_my_commands([
        BotCommand("start",     "Bắt đầu / Chào mừng"),
        BotCommand("help",      "Xem tất cả lệnh"),
        BotCommand("clear",     "Xóa lịch sử hội thoại"),
        BotCommand("model",     "Chọn AI model"),
        BotCommand("models",    "Danh sách tất cả models"),
        BotCommand("auto",      "Bật/tắt tự động chọn model"),
        BotCommand("profile",   "Xem/cài hồ sơ cá nhân"),
        BotCommand("stats",     "Thống kê chat"),
        BotCommand("remind",    "Đặt nhắc nhở"),
        BotCommand("reminders", "Danh sách nhắc nhở"),
        BotCommand("mn",        "Quản lý tài chính"),
        BotCommand("pro",       "Phân tích sâu"),
        BotCommand("agent",     "Agentic AI loop"),
        BotCommand("coder",     "Workflow lập trình"),
        BotCommand("rag",       "Quản lý tài liệu RAG"),
        BotCommand("tokens",    "Thống kê token và chi phí"),
    ])
    logger.info("Database initialized. Commands registered. Reminder loop started.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    # Commands
    app.add_handler(CommandHandler("start",     cmd_start))
    app.add_handler(CommandHandler("help",      cmd_help))
    app.add_handler(CommandHandler("clear",     cmd_clear))
    app.add_handler(CommandHandler("model",     cmd_model))
    app.add_handler(CommandHandler("models",    cmd_models))
    app.add_handler(CommandHandler("auto",      cmd_auto))
    app.add_handler(CommandHandler("profile",   cmd_profile))
    app.add_handler(CommandHandler("stats",     cmd_stats))
    app.add_handler(CommandHandler("remind",    cmd_remind))
    app.add_handler(CommandHandler("reminders", cmd_reminders))
    app.add_handler(CommandHandler("mn",        cmd_mn))
    app.add_handler(CommandHandler("pro",       cmd_pro))
    app.add_handler(CommandHandler("agent",     cmd_agent))
    app.add_handler(CommandHandler("coder",     cmd_coder))
    app.add_handler(CommandHandler("rag",       cmd_rag))
    app.add_handler(CommandHandler("tokens",    cmd_tokens))

    # Message handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Callback queries (single handler covers all patterns)
    app.add_handler(CallbackQueryHandler(handle_callback))

    # Error handler
    app.add_error_handler(error_handler)

    logger.info("Ultra Bolt bot starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
