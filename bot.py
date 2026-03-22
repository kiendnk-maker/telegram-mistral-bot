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
from telegram import Update, BotCommand, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)
from telegram.constants import ChatAction, ParseMode
from telegram.error import TelegramError

from database import init_db
from money_tracker import handle_money_command
from llm_core import call_llm, call_llm_stream, call_vision, transcribe_audio
from rag_core import has_docs, add_document, build_rag_context
from reminder_system import reminder_loop
from command_handler import (
    cmd_start, cmd_help, cmd_clear, cmd_model, cmd_models,
    cmd_auto, cmd_profile, cmd_stats, cmd_remind, cmd_reminders,
    cmd_mn, cmd_pro, cmd_agent, cmd_coder, cmd_rag, cmd_tokens,
    cmd_todo, cmd_tasks, cmd_done, cmd_deltask,
    cmd_pomodoro, cmd_motivation, cmd_checkin,
    cmd_user,
    handle_callback,
)
from prompts import MODEL_REGISTRY

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("Thiếu TELEGRAM_TOKEN trong file .env")
if not os.getenv("MISTRAL_API_KEY"):
    raise RuntimeError("Thiếu MISTRAL_API_KEY trong file .env")

# ── Owner-only access ─────────────────────────────────────────────────────────
_OWNER_ID_STR = os.getenv("OWNER_ID", "")
OWNER_ID: int = int(_OWNER_ID_STR) if _OWNER_ID_STR.strip().isdigit() else 0


# ── Retry model menu ──────────────────────────────────────────────────────────

RETRY_MODELS = [
    ("small",      "🔹 Mistral S"),
    ("large",      "🔵 Mistral L"),
    ("qwen3",      "🌟 Qwen3"),
    ("gpt_120b",   "🧠 GPT 120B"),
    ("groq_large", "🦙 Llama 70B"),
]


def _retry_keyboard(current_key: str) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(name, callback_data=f"retry_{key}")
        for key, name in RETRY_MODELS
        if key != current_key
    ]
    rows = [buttons[i:i+3] for i in range(0, len(buttons), 3)]
    return InlineKeyboardMarkup(rows)


async def _is_authorized(user_id: int) -> bool:
    """Owner always authorized; others checked against DB whitelist."""
    if OWNER_ID == 0:
        return True  # no restriction if OWNER_ID not set
    if user_id == OWNER_ID:
        return True
    from database import is_user_allowed
    return await is_user_allowed(user_id)


def _auth(fn):
    """Decorator: reject unauthorized users before running any command/handler."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id if update.effective_user else 0
        if not await _is_authorized(user_id):
            logger.warning(f"Unauthorized access attempt from user_id={user_id}")
            if update.effective_message:
                await update.effective_message.reply_text("⛔ Bot này là riêng tư.")
            return
        return await fn(update, context)
    wrapper.__name__ = fn.__name__
    return wrapper

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
        [KeyboardButton("📋 Công việc"),   KeyboardButton("🍅 Pomodoro")],
        [KeyboardButton("⚙️ Cài đặt"),    KeyboardButton("❓ Trợ giúp")],
    ],
    resize_keyboard=True,
    is_persistent=True,
)

# Set of menu button texts for fast lookup
MENU_BUTTONS = {
    "💬 Chat mới",
    "🤖 Đổi Model",
    "💰 Chi tiêu",
    "⏰ Nhắc nhở",
    "📚 Tài liệu",
    "📊 Thống kê",
    "📋 Công việc",
    "🍅 Pomodoro",
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


_TG_TAG = re.compile(
    r'<(/?)(?:b|i|u|s|code|pre|blockquote|tg-spoiler)(?:\s[^>]*)?>|<a\s[^>]*href=[^>]*>|</a>',
    re.IGNORECASE,
)


def _md_to_html(text: str) -> str:
    """
    Convert model output (markdown or HTML) to Telegram-safe HTML.
    - Preserves valid Telegram tags (b, i, u, s, code, pre, blockquote, a)
    - Converts unsupported HTML (<ul><li> → bullets, <br> → newline, etc.)
    - Converts markdown syntax (**, ##, *, __)
    """
    code_blocks: list[str] = []
    inline_codes: list[str] = []
    saved_tags: list[str] = []

    def save_code_block(m):
        code_blocks.append(html.escape(m.group(1)))
        return f"%%CB{len(code_blocks) - 1}%%"

    def save_inline_code(m):
        inline_codes.append(html.escape(m.group(1)))
        return f"%%IC{len(inline_codes) - 1}%%"

    def save_tg_tag(m):
        saved_tags.append(m.group(0))
        return f"%%TG{len(saved_tags) - 1}%%"

    # 1. Save code blocks before anything
    text = re.sub(r'```[\w]*\n?([\s\S]+?)```', save_code_block, text)
    text = re.sub(r'`([^`\n]+)`', save_inline_code, text)

    # 2. Convert unsupported HTML tags to text equivalents
    text = re.sub(r'<li[^>]*>', '• ', text, flags=re.IGNORECASE)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<hr\s*/?>', '─────────────────', text, flags=re.IGNORECASE)
    text = re.sub(r'</?(?:ul|ol|p|div|span|h[1-6])[^>]*>', '\n', text, flags=re.IGNORECASE)
    # Strip any remaining unknown tags
    text = re.sub(r'</?[a-zA-Z][^>]{0,50}>', '', text)

    # 3. Save valid Telegram tags before html.escape
    text = _TG_TAG.sub(save_tg_tag, text)

    # 4. Escape remaining special chars
    text = html.escape(text)

    # 5. Restore Telegram tags
    for i, tag in enumerate(saved_tags):
        text = text.replace(f"%%TG{i}%%", tag)

    # 6. Markdown conversions (only if not already HTML-formatted)
    # Horizontal rules
    text = re.sub(r'^[ \t]*[-*_]{3,}[ \t]*$', '─────────────────', text, flags=re.MULTILINE)
    # Headings → bold
    text = re.sub(r'^#{1,6}\s+(.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    # Bold **text**
    text = re.sub(r'\*\*([\s\S]+?)\*\*', r'<b>\1</b>', text)
    # Italic *text*
    text = re.sub(r'\*([^*\n]+?)\*', r'<i>\1</i>', text)
    # Underline __text__
    text = re.sub(r'__([\s\S]+?)__', r'<u>\1</u>', text)
    # Markdown list markers → bullet
    text = re.sub(r'^[ \t]*[-•]\s+', '• ', text, flags=re.MULTILINE)
    # Numbered list indent fix
    text = re.sub(r'^[ \t]+(\d+\.)', r'\1', text, flags=re.MULTILINE)

    # 7. Restore code
    for i, code in enumerate(inline_codes):
        text = text.replace(f"%%IC{i}%%", f"<code>{code}</code>")
    for i, code in enumerate(code_blocks):
        text = text.replace(f"%%CB{i}%%", f"<pre>{code}</pre>")

    return text


def _strip_thinking(text: str) -> tuple[str, bool]:
    """Remove <think>...</think> blocks. Returns (clean_text, had_thinking)."""
    import re as _re
    think_pattern = _re.compile(r'<think>[\s\S]*?</think>', _re.IGNORECASE)
    had = bool(think_pattern.search(text))
    clean = think_pattern.sub('', text).strip()
    return clean, had


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
    if not await _is_authorized(user_id):
        await update.message.reply_text("⛔ Bot này là riêng tư.")
        return
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
        elif user_message == "📋 Công việc":
            await cmd_tasks(update, context)
        elif user_message == "🍅 Pomodoro":
            await cmd_pomodoro(update, context)
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

        # ── Streaming response ────────────────────────────────────────────────
        sent_msg = await update.message.reply_html("⌛")
        full_text = ""
        model_key = "groq_large"
        last_edit = 0.0
        EDIT_INTERVAL = 1.2  # seconds between edits (safe Telegram rate limit)

        async for chunk, mk in call_llm_stream(user_id, user_message, extra_context=extra_context):
            full_text += chunk
            model_key = mk
            now = time.time()
            if now - last_edit >= EDIT_INTERVAL and full_text.strip():
                try:
                    clean_preview, _ = _strip_thinking(full_text)
                    preview = _md_to_html(clean_preview or full_text)
                    if len(preview) < 3800:
                        await sent_msg.edit_text(preview + " ▌", parse_mode=ParseMode.HTML)
                    last_edit = now
                except Exception:
                    pass

        elapsed = round(time.time() - start_time, 1)
        clean_text, had_thinking = _strip_thinking(full_text)
        reply_html = _md_to_html(clean_text)
        model_name = MODEL_REGISTRY.get(model_key, {}).get("name", model_key)
        rag_badge = " 📚" if extra_context else ""
        think_badge = " 🧠" if had_thinking else ""
        footer = f"\n\n<i>⏱ {elapsed}s · {model_name}{rag_badge}{think_badge}</i>"

        # Store for retry
        context.user_data["last_msg"] = user_message
        context.user_data["last_extra"] = extra_context

        retry_kb = _retry_keyboard(model_key)
        full_out = reply_html + footer
        if len(full_out) <= 4000:
            await sent_msg.edit_text(full_out, parse_mode=ParseMode.HTML, reply_markup=retry_kb)
        else:
            await sent_msg.delete()
            await _send_long(update, full_out, reply_markup=retry_kb)

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
    model_key = await get_setting(user_id, "model_key", "groq_large")
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
    if not await _is_authorized(user_id):
        await update.message.reply_text("⛔ Bot này là riêng tư.")
        return

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
    if not await _is_authorized(user_id):
        await update.message.reply_text("⛔ Bot này là riêng tư.")
        return

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
    if not await _is_authorized(user_id):
        await update.message.reply_text("⛔ Bot này là riêng tư.")
        return
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


# ── Retry with different model ────────────────────────────────────────────────

async def handle_retry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if not await _is_authorized(user_id):
        return

    model_key = query.data[len("retry_"):]
    if model_key not in MODEL_REGISTRY:
        return

    last_msg = context.user_data.get("last_msg")
    last_extra = context.user_data.get("last_extra")
    if not last_msg:
        await query.answer("Không tìm thấy câu hỏi gốc.", show_alert=True)
        return

    model_name = MODEL_REGISTRY[model_key]["name"]
    sent = await query.message.reply_html(f"⌛ <i>{html.escape(model_name)}...</i>")

    start_time = time.time()
    full_text = ""
    last_edit = 0.0

    try:
        async for chunk, mk in call_llm_stream(
            user_id, last_msg,
            model_key=model_key,
            extra_context=last_extra,
            save_history=False,
        ):
            full_text += chunk
            now = time.time()
            if now - last_edit >= 1.2 and full_text.strip():
                try:
                    preview = _md_to_html(full_text)
                    if len(preview) < 3800:
                        await sent.edit_text(preview + " ▌", parse_mode=ParseMode.HTML)
                    last_edit = now
                except Exception:
                    pass

        elapsed = round(time.time() - start_time, 1)
        clean_text, had_thinking = _strip_thinking(full_text)
        reply_html = _md_to_html(clean_text)
        rag_badge = " 📚" if last_extra else ""
        think_badge = " 🧠" if had_thinking else ""
        footer = f"\n\n<i>⏱ {elapsed}s · {html.escape(model_name)}{rag_badge}{think_badge}</i>"
        full_out = reply_html + footer

        retry_kb = _retry_keyboard(model_key)
        if len(full_out) <= 4000:
            await sent.edit_text(full_out, parse_mode=ParseMode.HTML, reply_markup=retry_kb)
        else:
            await sent.delete()
            parts = [full_out[i:i+4000] for i in range(0, len(full_out), 4000)]
            for idx, part in enumerate(parts):
                markup = retry_kb if idx == len(parts) - 1 else None
                await query.message.reply_html(part, reply_markup=markup)

    except Exception as e:
        logger.error(f"Retry error user {user_id}: {e}", exc_info=True)
        await sent.edit_text("⚠️ Lỗi khi thử lại. Vui lòng thử lại.")


# ── Unauthorized access handler ───────────────────────────────────────────────

async def handle_unauthorized(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    logger.warning(f"Unauthorized access attempt from user_id={user.id if user else '?'}")
    if update.effective_message:
        await update.effective_message.reply_text("⛔ Bot này là riêng tư.")


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
        BotCommand("rag",        "Quản lý tài liệu RAG"),
        BotCommand("tokens",     "Thống kê token và chi phí"),
        BotCommand("todo",       "Thêm việc cần làm"),
        BotCommand("tasks",      "Xem danh sách việc cần làm"),
        BotCommand("done",       "Đánh dấu hoàn thành"),
        BotCommand("deltask",    "Xóa task"),
        BotCommand("pomodoro",   "Bắt đầu Pomodoro 25 phút"),
        BotCommand("motivation", "Nhận câu động lực"),
        BotCommand("checkin",    "Tổng kết ngày hôm nay"),
        BotCommand("user",       "Quản lý danh sách người dùng"),
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

    # Commands (auth enforced via _auth wrapper)
    _cmd = lambda name, fn: app.add_handler(CommandHandler(name, _auth(fn)))
    _cmd("start",     cmd_start)
    _cmd("help",      cmd_help)
    _cmd("clear",     cmd_clear)
    _cmd("model",     cmd_model)
    _cmd("models",    cmd_models)
    _cmd("auto",      cmd_auto)
    _cmd("profile",   cmd_profile)
    _cmd("stats",     cmd_stats)
    _cmd("remind",    cmd_remind)
    _cmd("reminders", cmd_reminders)
    _cmd("mn",        cmd_mn)
    _cmd("pro",       cmd_pro)
    _cmd("agent",     cmd_agent)
    _cmd("coder",     cmd_coder)
    _cmd("rag",       cmd_rag)
    _cmd("tokens",    cmd_tokens)
    _cmd("todo",      cmd_todo)
    _cmd("tasks",     cmd_tasks)
    _cmd("done",      cmd_done)
    _cmd("deltask",   cmd_deltask)
    _cmd("pomodoro",  cmd_pomodoro)
    _cmd("motivation",cmd_motivation)
    _cmd("checkin",   cmd_checkin)
    _cmd("user",      cmd_user)

    # Message handlers (auth check inside each handler)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Retry handler (pattern match first, then general callbacks)
    app.add_handler(CallbackQueryHandler(handle_retry, pattern=r"^retry_"))
    app.add_handler(CallbackQueryHandler(handle_callback, block=False))

    # Error handler
    app.add_error_handler(error_handler)

    logger.info("Ultra Bolt bot starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
