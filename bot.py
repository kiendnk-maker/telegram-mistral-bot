import os
import re
import html
import time
import logging
import asyncio
from collections import defaultdict
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)
from telegram.constants import ChatAction, ParseMode
from telegram.error import TelegramError
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Validate API keys ngay khi khởi động
if not TELEGRAM_TOKEN or not MISTRAL_API_KEY:
    raise RuntimeError("Thiếu TELEGRAM_TOKEN hoặc MISTRAL_API_KEY trong file .env")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

mistral = MistralClient(api_key=MISTRAL_API_KEY)

# Config
MODEL = "mistral-small-latest"
MAX_MSG_LENGTH = 2000
MAX_HISTORY = 20
RATE_LIMIT = 10        # tin nhắn tối đa
RATE_WINDOW = 60       # trong N giây
TYPING_TIMEOUT = 30    # giây timeout

SYSTEM_PROMPT = """Bạn là trợ lý AI thông minh, trả lời ngắn gọn bằng tiếng Việt.
Định dạng câu trả lời dùng HTML Telegram:
- <b>text</b> cho chữ đậm
- <i>text</i> cho chữ nghiêng
- <code>text</code> cho code ngắn
- <pre>text</pre> cho code block
- Dùng emoji phù hợp, tự nhiên
- Không dùng ** hay ## markdown"""

# Lưu trữ trong memory
conversations: dict[int, list] = {}
rate_tracker: dict[int, list] = defaultdict(list)


def is_rate_limited(user_id: int) -> tuple[bool, int]:
    now = time.time()
    timestamps = rate_tracker[user_id]
    # Xóa timestamp cũ hơn RATE_WINDOW
    rate_tracker[user_id] = [t for t in timestamps if now - t < RATE_WINDOW]
    if len(rate_tracker[user_id]) >= RATE_LIMIT:
        wait = int(RATE_WINDOW - (now - rate_tracker[user_id][0])) + 1
        return True, wait
    rate_tracker[user_id].append(now)
    return False, 0


def md_to_html(text: str) -> str:
    """Chuyển markdown sang HTML Telegram an toàn."""
    # Tách code blocks ra trước để không bị ảnh hưởng
    code_blocks = []
    def save_code_block(m):
        code_blocks.append(html.escape(m.group(1)))
        return f"%%CODEBLOCK{len(code_blocks)-1}%%"

    inline_codes = []
    def save_inline_code(m):
        inline_codes.append(html.escape(m.group(1)))
        return f"%%INLINECODE{len(inline_codes)-1}%%"

    text = re.sub(r'```[\w]*\n?([\s\S]+?)```', save_code_block, text)
    text = re.sub(r'`([^`\n]+)`', save_inline_code, text)

    # Escape HTML
    text = html.escape(text)

    # Áp dụng format
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*([^*\n]+?)\*', r'<i>\1</i>', text)
    text = re.sub(r'__(.*?)__', r'<u>\1</u>', text)

    # Khôi phục code
    for i, code in enumerate(inline_codes):
        text = text.replace(f"%%INLINECODE{i}%%", f"<code>{code}</code>")
    for i, code in enumerate(code_blocks):
        text = text.replace(f"%%CODEBLOCK{i}%%", f"<pre>{code}</pre>")

    return text


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    name = update.effective_user.first_name or "bạn"
    conversations[user_id] = []
    await update.message.reply_html(
        f"👋 Xin chào <b>{html.escape(name)}</b>!\n\n"
        f"Tôi là chatbot powered by <b>Mistral AI</b> 🤖\n"
        f"Model: <code>{MODEL}</code>\n\n"
        "📌 <b>Lệnh:</b>\n"
        "/clear — Xóa lịch sử hội thoại\n"
        "/model — Đổi model\n"
        "/stats — Thống kê\n\n"
        "💬 Hãy hỏi tôi bất cứ điều gì!"
    )


async def clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
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


async def clear_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    if query.data == "clear_yes":
        conversations[user_id] = []
        await query.edit_message_text("🗑 <b>Đã xóa lịch sử hội thoại.</b>", parse_mode=ParseMode.HTML)
    else:
        await query.edit_message_text("↩️ Đã hủy.")


async def model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("⚡ mistral-small (nhanh)", callback_data="model_small")],
        [InlineKeyboardButton("🧠 mistral-large (thông minh)", callback_data="model_large")],
        [InlineKeyboardButton("❌ Hủy", callback_data="model_cancel")],
    ])
    await update.message.reply_html(
        f"🤖 Model hiện tại: <code>{MODEL}</code>\n\nChọn model muốn dùng:",
        reply_markup=keyboard
    )


async def model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global MODEL
    query = update.callback_query
    await query.answer()
    if query.data == "model_small":
        MODEL = "mistral-small-latest"
        await query.edit_message_text("✅ Đã chuyển sang <code>mistral-small-latest</code> ⚡ (nhanh)", parse_mode=ParseMode.HTML)
    elif query.data == "model_large":
        MODEL = "mistral-large-latest"
        await query.edit_message_text("✅ Đã chuyển sang <code>mistral-large-latest</code> 🧠 (thông minh hơn)", parse_mode=ParseMode.HTML)
    else:
        await query.edit_message_text("↩️ Đã hủy.")


async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    history = conversations.get(user_id, [])
    msg_count = len(history)
    user_msgs = sum(1 for m in history if m.role == "user")
    await update.message.reply_html(
        f"📊 <b>Thống kê của bạn:</b>\n\n"
        f"💬 Tin nhắn trong phiên: <b>{msg_count}</b>\n"
        f"👤 Câu hỏi của bạn: <b>{user_msgs}</b>\n"
        f"🤖 Phản hồi của bot: <b>{msg_count - user_msgs}</b>\n"
        f"🧠 Model đang dùng: <code>{MODEL}</code>"
    )


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_message = update.message.text

    # Kiểm tra độ dài tin nhắn
    if len(user_message) > MAX_MSG_LENGTH:
        await update.message.reply_html(
            f"⚠️ Tin nhắn quá dài (<b>{len(user_message)}/{MAX_MSG_LENGTH}</b> ký tự). "
            "Vui lòng rút gọn lại."
        )
        return

    # Rate limiting
    limited, wait_sec = is_rate_limited(user_id)
    if limited:
        await update.message.reply_html(
            f"⏳ Bạn gửi quá nhanh! Vui lòng chờ <b>{wait_sec} giây</b>."
        )
        return

    if user_id not in conversations:
        conversations[user_id] = []

    conversations[user_id].append(ChatMessage(role="user", content=user_message))

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    start_time = time.time()

    try:
        messages = [ChatMessage(role="system", content=SYSTEM_PROMPT)] + conversations[user_id]

        response = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None,
                lambda: mistral.chat(
                    model=MODEL,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.7,
                )
            ),
            timeout=TYPING_TIMEOUT
        )

        reply = response.choices[0].message.content
        elapsed = round(time.time() - start_time, 1)
        reply_html = md_to_html(reply)

        conversations[user_id].append(ChatMessage(role="assistant", content=reply))

        # Giới hạn lịch sử
        if len(conversations[user_id]) > MAX_HISTORY:
            conversations[user_id] = conversations[user_id][-MAX_HISTORY:]

        # Thêm footer nhỏ
        footer = f"\n\n<i>⏱ {elapsed}s · {MODEL.split('-')[1]}</i>"

        await update.message.reply_html(reply_html + footer)
        logger.info(f"User {user_id} | {elapsed}s | {len(user_message)} chars")

    except asyncio.TimeoutError:
        await update.message.reply_html(
            "⏰ <b>Yêu cầu mất quá nhiều thời gian.</b>\n"
            "Thử hỏi câu ngắn hơn hoặc dùng /model để chọn <code>mistral-small</code>."
        )
    except Exception as e:
        logger.error(f"Lỗi chat user {user_id}: {e}")
        await update.message.reply_html(
            "⚠️ <b>Có lỗi xảy ra.</b> Vui lòng thử lại sau.\n"
            "Nếu lỗi tiếp tục, dùng /clear để reset hội thoại."
        )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Lỗi hệ thống: {context.error}")


def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear_cmd))
    app.add_handler(CommandHandler("model", model_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))
    app.add_handler(CallbackQueryHandler(clear_callback, pattern="^clear_"))
    app.add_handler(CallbackQueryHandler(model_callback, pattern="^model_"))
    app.add_error_handler(error_handler)

    logger.info(f"🤖 Bot khởi động | Model: {MODEL}")
    app.run_polling()


if __name__ == "__main__":
    main()
