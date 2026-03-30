"""
bot.py - Main entry point for Ultra Bolt Telegram bot
"""
import os
import re
import html
import time
from datetime import datetime
import logging
import asyncio
import base64
import tempfile
from collections import defaultdict

from dotenv import load_dotenv
from telegram import Update, BotCommand, ReplyKeyboardRemove, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)
from telegram.constants import ChatAction, ParseMode
from telegram.error import TelegramError

from database import init_db
from money_tracker import handle_money_command
from llm_core import call_llm, call_llm_stream, call_vision_stream, call_ocr_mistral, transcribe_audio
from rag_core import has_docs, add_document, build_rag_context
from reminder_system import reminder_loop
from api_dashboard import cmd_mapi_telegram, cmd_gapi_telegram
from command_handler import (
    cmd_start, cmd_help, cmd_clear, cmd_model, cmd_models,
    cmd_auto, cmd_profile, cmd_stats, cmd_remind, cmd_reminders,
    cmd_mn, cmd_pro, cmd_agent, cmd_coder, cmd_rag, cmd_tokens,
    cmd_todo, cmd_tasks, cmd_done, cmd_deltask,
    cmd_pomodoro, cmd_motivation, cmd_checkin,
    cmd_user,
    cmd_tw, cmd_vi,
    handle_callback,
    cmd_web, cmd_sum, cmd_quiz,
    cmd_gauth, cmd_cal, cmd_gmail, cmd_gdrive,
)
from prompts import MODEL_REGISTRY

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("Thiếu TELEGRAM_TOKEN trong file .env")
if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("Thiếu GEMINI_API_KEY trong file .env")

# ── Owner-only access ─────────────────────────────────────────────────────────
_OWNER_ID_STR = os.getenv("OWNER_ID", "")
OWNER_ID: int = int(_OWNER_ID_STR) if _OWNER_ID_STR.strip().isdigit() else 0


# ── Retry model menu ──────────────────────────────────────────────────────────

RETRY_MODELS = [
    ("flash",       "⚡ Gemini Flash"),
    ("flash_lite",  "💨 Flash Lite"),
    ("flash_think", "💭 Flash Think"),
    ("pro",         "🧠 Gemini Pro"),
]


def _retry_keyboard(current_key: str) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(name, callback_data=f"retry_{key}")
        for key, name in RETRY_MODELS
        if key != current_key
    ]
    rows = [buttons[i:i+3] for i in range(0, len(buttons), 3)]
    return InlineKeyboardMarkup(rows)


# ── Vision model menu ─────────────────────────────────────────────────────────

VISION_FOLLOWUP_MODELS = [
    ("flash",       "👁 Gemini Flash"),
    ("flash_think", "💭 Flash Think"),
    ("pro",         "🧠 Gemini Pro"),
]


def _vision_choice_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("🔤 OCR - Trích xuất chữ", callback_data="vision_ocr"),
        InlineKeyboardButton("🖼 Mô tả ảnh", callback_data="vision_describe"),
    ]])


def _ocr_followup_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("🖼 Mô tả ảnh", callback_data="vision_describe"),
        InlineKeyboardButton("❌ Xóa ảnh", callback_data="clear_vision"),
    ]])


def _vision_keyboard(current_key: str = "flash") -> InlineKeyboardMarkup:
    model_btns = [
        InlineKeyboardButton(name, callback_data=f"vmodel_{key}")
        for key, name in VISION_FOLLOWUP_MODELS
        if key != current_key
    ]
    rows = [model_btns[i:i+3] for i in range(0, len(model_btns), 3)]
    rows.insert(0, [InlineKeyboardButton("🔤 OCR", callback_data="vision_ocr")])
    rows.append([InlineKeyboardButton("❌ Xóa ảnh", callback_data="clear_vision")])
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

_rate_tracker: dict[int, list[float]] = defaultdict(list)

MAIN_MENU = ReplyKeyboardRemove()
MENU_BUTTONS: set = set()


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
            for key in ("vision_image", "vision_image_hq", "vision_caption", "vision_messages", "vision_model", "vision_desc", "vision_mode"):
                context.user_data.pop(key, None)
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

    # ── Quiz answer detection ─────────────────────────────────────────────────
    from web_tools import check_quiz_answer, has_active_quiz
    if has_active_quiz(user_id) and len(user_message.strip()) <= 2:
        result = check_quiz_answer(user_id, user_message.strip())
        if result:
            await update.message.reply_html(result)
            return

    # ── Normal LLM flow ──────────────────────────────────────────────────────
    limited, wait_sec = _is_rate_limited(user_id)
    if limited:
        await update.message.reply_html(
            f"⏳ Bạn gửi quá nhanh! Vui lòng chờ <b>{wait_sec} giây</b>."
        )
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    start_time = time.time()

    # ── Vision follow-up: user is asking about a stored image ─────────────────
    vision_msgs = context.user_data.get("vision_messages")
    if vision_msgs is not None:
        vision_model = context.user_data.get("vision_model", "flash")
        vision_mode = context.user_data.get("vision_mode", "describe")
        vision_desc = context.user_data.get("vision_desc", "")
        try:
            sent_msg = await update.message.reply_html("⌛")
            full_text = ""
            model_key = vision_model
            last_edit = 0.0

            if vision_mode == "ocr":
                # OCR follow-up: text model explains the extracted text
                extra_context = f"Nội dung OCR trích xuất từ ảnh:\n{vision_desc}"
                async for chunk, mk in call_llm_stream(
                    user_id, user_message, extra_context=extra_context,
                ):
                    full_text += chunk
                    model_key = mk
                    now = time.time()
                    if now - last_edit >= 1.2 and full_text.strip():
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
                think_badge = " 🧠" if had_thinking else ""
                footer = f"\n\n<i>⏱ {elapsed}s · {model_name} · 🔤 OCR context{think_badge}</i>"
                kb = _ocr_followup_keyboard()

            elif vision_model == "flash":
                # Describe follow-up: continue multi-turn with actual image
                vision_msgs.append({"role": "user", "content": user_message})
                async for chunk, mk in call_vision_stream(user_id, vision_msgs):
                    full_text += chunk
                    model_key = mk
                    now = time.time()
                    if now - last_edit >= 1.2 and full_text.strip():
                        try:
                            preview = _md_to_html(full_text)
                            if len(preview) < 3800:
                                await sent_msg.edit_text(preview + " ▌", parse_mode=ParseMode.HTML)
                            last_edit = now
                        except Exception:
                            pass
                vision_msgs.append({"role": "assistant", "content": full_text})
                context.user_data["vision_messages"] = vision_msgs
                elapsed = round(time.time() - start_time, 1)
                reply_html = _md_to_html(full_text)
                model_name = MODEL_REGISTRY.get(model_key, {}).get("name", model_key)
                footer = f"\n\n<i>⏱ {elapsed}s · {model_name} · 🖼 hỏi về ảnh</i>"
                kb = _vision_keyboard(model_key)

            else:
                # Text model with vision description as context
                extra_context = f"Mô tả ảnh (phân tích bởi AI vision):\n{vision_desc}"
                async for chunk, mk in call_llm_stream(
                    user_id, user_message,
                    model_key=vision_model,
                    extra_context=extra_context,
                ):
                    full_text += chunk
                    model_key = mk
                    now = time.time()
                    if now - last_edit >= 1.2 and full_text.strip():
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
                think_badge = " 🧠" if had_thinking else ""
                footer = f"\n\n<i>⏱ {elapsed}s · {model_name} · 🖼 hỏi về ảnh{think_badge}</i>"
                kb = _vision_keyboard(model_key)

            full_out = reply_html + footer
            if len(full_out) <= 4000:
                await sent_msg.edit_text(full_out, parse_mode=ParseMode.HTML, reply_markup=kb)
            else:
                await sent_msg.delete()
                await _send_long(update, full_out, reply_markup=kb)
            logger.info(f"User {user_id} | vision follow-up ({vision_mode}) | {elapsed}s | model={model_key}")
        except Exception as e:
            err_str = str(e).lower()
            if "429" in str(e) or "rate limit" in err_str or "rate_limited" in err_str:
                await update.message.reply_html(
                    "⏳ <b>Gemini Vision đang bị rate limit.</b>\nChờ 30-60 giây rồi hỏi lại."
                )
            else:
                logger.error(f"Vision follow-up error user {user_id}: {e}", exc_info=True)
                await update.message.reply_html("⚠️ Lỗi hỏi về ảnh. Vui lòng thử lại.")
        return

    # ── Standard LLM flow ─────────────────────────────────────────────────────
    try:
        # Check for RAG context
        extra_context = None
        if await has_docs(user_id):
            extra_context = await build_rag_context(user_id, user_message)

        # ── Remove persistent keyboard (one-time signal, separate message) ──────
        try:
            rm = await update.message.reply_text("\u200b", reply_markup=ReplyKeyboardRemove())
            await rm.delete()
        except Exception:
            pass

        # ── Streaming response ────────────────────────────────────────────────
        sent_msg = await update.message.reply_html("⌛")
        full_text = ""
        model_key = "flash"
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
            "Thử câu ngắn hơn hoặc dùng /model để chọn Gemini Flash Lite."
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
    model_key = await get_setting(user_id, "model_key", "flash")
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

    if not os.getenv("GEMINI_API_KEY"):
        await update.message.reply_html(
            "⚠️ Tính năng giọng nói chưa được cấu hình (thiếu GEMINI_API_KEY)."
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

        # Fix 1: language follows /vi /tw setting
        from database import get_setting as _get_setting
        lang_mode = await _get_setting(user_id, "lang_mode", "vi")
        whisper_lang = "zh" if lang_mode == "zh-TW" else "vi"

        transcript = await transcribe_audio(tmp_path, language=whisper_lang)
        if not transcript or not transcript.strip():
            await update.message.reply_html("🎙 Không nhận diện được giọng nói. Vui lòng thử lại.")
            return

        await update.message.reply_html(f"🎙 <i>Đã nghe: {html.escape(transcript)}</i>")
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

        extra_context = None
        if await has_docs(user_id):
            extra_context = await build_rag_context(user_id, transcript)

        # Fix 2: streaming response with live editing
        sent_msg = await update.message.reply_html("⌛")
        full_text = ""
        model_key = "flash"
        last_edit = 0.0
        start_time = time.time()

        async for chunk, mk in call_llm_stream(user_id, transcript, extra_context=extra_context):
            full_text += chunk
            model_key = mk
            now = time.time()
            if now - last_edit >= 1.2 and full_text.strip():
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
        think_badge = " 🧠" if had_thinking else ""
        footer = f"\n\n<i>⏱ {elapsed}s · 🎙 STT · {model_name}{think_badge}</i>"

        # Fix 3: store for retry + retry keyboard
        context.user_data["last_msg"] = transcript
        context.user_data["last_extra"] = extra_context
        retry_kb = _retry_keyboard(model_key)

        full_out = reply_html + footer
        if len(full_out) <= 4000:
            await sent_msg.edit_text(full_out, parse_mode=ParseMode.HTML, reply_markup=retry_kb)
        else:
            await sent_msg.delete()
            await _send_long(update, full_out, reply_markup=retry_kb)

        logger.info(f"User {user_id} | voice | {elapsed}s | model={model_key} | lang={whisper_lang}")

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

        from PIL import Image
        import io
        img = Image.open(tmp_path)

        # 512px for vision (token-efficient)
        img_small = img.copy()
        img_small.thumbnail((512, 512), Image.LANCZOS)
        buf = io.BytesIO()
        img_small.save(buf, format="JPEG", quality=80)
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # 1024px for OCR (better text accuracy)
        img_hq = img.copy()
        img_hq.thumbnail((1024, 1024), Image.LANCZOS)
        buf_hq = io.BytesIO()
        img_hq.save(buf_hq, format="JPEG", quality=90)
        image_b64_hq = base64.b64encode(buf_hq.getvalue()).decode("utf-8")

        # Save image state, clear previous vision session
        context.user_data["vision_image"] = image_b64
        context.user_data["vision_image_hq"] = image_b64_hq  # 1024px for OCR
        context.user_data["vision_caption"] = update.message.caption or ""
        context.user_data["vision_messages"] = None
        context.user_data["vision_mode"] = None
        context.user_data["vision_desc"] = None
        context.user_data["vision_model"] = "flash"

        await update.message.reply_html(
            "📸 <b>Ảnh đã nhận!</b> Bạn muốn làm gì?",
            reply_markup=_vision_choice_keyboard(),
        )
        logger.info(f"User {user_id} | photo received | saved for vision choice")

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


# ── Vision model switch callback ──────────────────────────────────────────────

async def handle_vision_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Switch model used for follow-up questions about the stored image."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if not await _is_authorized(user_id):
        return

    model_key = query.data[len("vmodel_"):]
    if model_key not in MODEL_REGISTRY:
        return

    if context.user_data.get("vision_messages") is None:
        await query.answer("Không có ảnh nào đang được lưu.", show_alert=True)
        return

    context.user_data["vision_model"] = model_key
    model_name = MODEL_REGISTRY[model_key]["name"]

    if model_key == "flash":
        mode_text = "👁 Vision mode — hỏi trực tiếp với ảnh"
    else:
        mode_text = f"📝 Text mode — dùng mô tả ảnh làm context"

    await query.message.reply_html(
        f"✅ Đã chuyển sang <b>{html.escape(model_name)}</b>\n"
        f"<i>{mode_text}</i>\n\n"
        f"Gõ câu hỏi về ảnh:",
        reply_markup=_vision_keyboard(model_key),
    )


async def handle_clear_vision(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear stored image context."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if not await _is_authorized(user_id):
        return

    for key in ("vision_image", "vision_image_hq", "vision_caption", "vision_messages", "vision_model", "vision_desc", "vision_mode"):
        context.user_data.pop(key, None)

    await query.message.reply_html("🗑 Đã xóa ảnh. Chat bình thường tiếp tục.")


async def handle_vision_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle OCR / Describe choice after photo is sent."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if not await _is_authorized(user_id):
        return

    image_b64 = context.user_data.get("vision_image")
    if not image_b64:
        await query.answer("⚠️ Ảnh đã hết hạn, gửi lại ảnh nhé!", show_alert=True)
        return

    mode = query.data  # "vision_ocr" or "vision_describe"

    sent = await query.message.reply_html(
        "⌛ 🔤 Gemini OCR đang xử lý..." if mode == "vision_ocr" else "⌛ 🖼 Đang mô tả ảnh..."
    )
    start_time = time.time()

    try:
        if mode == "vision_ocr":
            # ── Gemini OCR ────────────────────────────────────────────────────
            image_b64_hq = context.user_data.get("vision_image_hq", image_b64)
            full_text = await call_ocr_mistral(user_id, image_b64_hq)
            elapsed = round(time.time() - start_time, 1)
            reply_html = _md_to_html(full_text)
            context.user_data["vision_messages"] = []   # no multi-turn for OCR
            context.user_data["vision_model"] = "flash"
            context.user_data["vision_desc"] = full_text
            context.user_data["vision_mode"] = "ocr"
            footer = f"\n\n<i>⏱ {elapsed}s · gemini-2.5-flash 🔤 · Gõ câu hỏi về nội dung</i>"
            kb = _ocr_followup_keyboard()

        else:
            # ── Gemini Vision describe ────────────────────────────────────────
            from database import get_setting as _get_setting
            lang_mode = await _get_setting(user_id, "lang_mode", "vi")
            caption = context.user_data.get("vision_caption", "")
            extra = f" {caption}" if caption else ""
            if lang_mode == "zh-TW":
                prompt = f"請用繁體中文詳細描述這張圖片的所有內容。{extra}"
            else:
                prompt = f"Mô tả chi tiết hình ảnh này bằng tiếng Việt.{extra}"

            vision_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            full_text = ""
            model_key = "flash"
            last_edit = 0.0

            async for chunk, mk in call_vision_stream(user_id, vision_messages):
                full_text += chunk
                model_key = mk
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
            reply_html = _md_to_html(full_text)
            model_name = MODEL_REGISTRY.get(model_key, {}).get("name", model_key)
            vision_messages.append({"role": "assistant", "content": full_text})
            context.user_data["vision_messages"] = vision_messages
            context.user_data["vision_model"] = "flash"
            context.user_data["vision_desc"] = full_text
            context.user_data["vision_mode"] = "describe"
            footer = f"\n\n<i>⏱ {elapsed}s · {model_name} · 🖼 Gõ tiếp để hỏi về ảnh</i>"
            kb = _vision_keyboard("flash")

        full_out = reply_html + footer
        if len(full_out) <= 4000:
            await sent.edit_text(full_out, parse_mode=ParseMode.HTML, reply_markup=kb)
        else:
            await sent.delete()
            await query.message.reply_html(full_out[:4000], reply_markup=kb)

        logger.info(f"User {user_id} | {mode} | {elapsed}s")

    except Exception as e:
        err_str = str(e).lower()
        if "429" in str(e) or "rate limit" in err_str or "rate_limited" in err_str:
            await sent.edit_text(
                "⏳ <b>Rate limit.</b> Chờ 30-60 giây rồi thử lại.",
                parse_mode=ParseMode.HTML,
            )
        else:
            logger.error(f"Vision choice error user {user_id}: {e}", exc_info=True)
            await sent.edit_text("⚠️ Lỗi xử lý ảnh. Vui lòng thử lại.")


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

    # Seed allowed_users from ALLOWED_USERS env var so whitelist survives redeploys.
    # Set ALLOWED_USERS=id1,id2,id3 in Railway / Render environment variables.
    from database import add_allowed_user as _add_user
    _allowed_env = os.getenv("ALLOWED_USERS", "")
    if _allowed_env:
        for _uid in _allowed_env.split(","):
            _uid = _uid.strip()
            if _uid.lstrip("-").isdigit():
                await _add_user(int(_uid), OWNER_ID or 0)

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
        BotCommand("tw",         "Chế độ trả lời Tiếng Trung phồn thể"),
        BotCommand("vi",         "Chế độ trả lời Tiếng Việt"),
        BotCommand("mapi",       "Mistral AI dashboard"),
        BotCommand("gapi",       "Groq Cloud dashboard"),
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
    _cmd("tw",        cmd_tw)
    _cmd("vi",        cmd_vi)
    _cmd("mapi",      cmd_mapi_telegram)
    _cmd("gapi",      cmd_gapi_telegram)
    _cmd("web",       cmd_web)
    _cmd("sum",       cmd_sum)
    _cmd("quiz",      cmd_quiz)
    _cmd("gauth",     cmd_gauth)
    _cmd("cal",       cmd_cal)
    _cmd("gmail",     cmd_gmail)
    _cmd("gdrive",    cmd_gdrive)

    # Message handlers (auth check inside each handler)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Retry handler (pattern match first, then general callbacks)
    app.add_handler(CallbackQueryHandler(handle_retry, pattern=r"^retry_"))
    app.add_handler(CallbackQueryHandler(handle_vision_choice, pattern=r"^vision_(ocr|describe)$"))
    app.add_handler(CallbackQueryHandler(handle_vision_model, pattern=r"^vmodel_"))
    app.add_handler(CallbackQueryHandler(handle_clear_vision, pattern=r"^clear_vision$"))
    app.add_handler(CallbackQueryHandler(handle_callback, block=False))

    # Error handler
    app.add_error_handler(error_handler)

    logger.info("Ultra Bolt bot starting...")

    # Run Telegram polling + OAuth web server together
    import asyncio
    from aiohttp import web as aio_web
    from oauth_server import create_oauth_app

    async def run_all():
        # Initialize telegram
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        logger.info("✅ Telegram polling started")

        # Start OAuth web server
        port = int(os.getenv("PORT", "8080"))
        oauth_app = create_oauth_app(telegram_bot=app.bot)
        runner = aio_web.AppRunner(oauth_app)
        await runner.setup()
        site = aio_web.TCPSite(runner, "0.0.0.0", port)
        await site.start()
        logger.info(f"✅ OAuth web server on port {port}")

        # Keep running forever
        try:
            await asyncio.Event().wait()
        finally:
            await app.updater.stop()
            await app.stop()
            await app.shutdown()
            await runner.cleanup()

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
