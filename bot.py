import os
import re
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode, ChatAction
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.WARNING
)

mistral = MistralClient(api_key=MISTRAL_API_KEY)

conversations: dict[int, list] = {}

SYSTEM_PROMPT = """Bạn là trợ lý AI thông minh, trả lời bằng tiếng Việt.
Khi định dạng câu trả lời, dùng HTML của Telegram:
- <b>text</b> cho chữ đậm (thay vì **)
- <i>text</i> cho chữ nghiêng
- <code>text</code> cho code ngắn
- <pre>text</pre> cho code block dài
- Dùng emoji phù hợp để trực quan hơn
- Trả lời ngắn gọn, súc tích, không dài dòng"""

def md_to_html(text: str) -> str:
    """Chuyển markdown ** sang HTML <b>, * sang <i>"""
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    text = re.sub(r'```[\w]*\n?([\s\S]+?)```', r'<pre>\1</pre>', text)
    # Escape ký tự đặc biệt HTML ngoài các tag đã dùng
    return text

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conversations[user_id] = []
    await update.message.reply_html(
        "👋 Xin chào! Tôi là chatbot powered by <b>Mistral AI</b>.\n\n"
        "💬 Hãy hỏi tôi bất cứ điều gì!\n\n"
        "📌 <b>Lệnh:</b>\n"
        "/start — Bắt đầu / Reset hội thoại\n"
        "/clear — Xóa lịch sử chat\n"
        "/model — Xem model đang dùng"
    )

async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conversations[user_id] = []
    await update.message.reply_html("🗑 <b>Đã xóa lịch sử hội thoại.</b>")

async def model_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_html(
        "🤖 <b>Model hiện tại:</b> <code>mistral-small-latest</code>\n"
        "⚡ Tốc độ nhanh, chất lượng tốt"
    )

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_message = update.message.text

    if user_id not in conversations:
        conversations[user_id] = []

    conversations[user_id].append(
        ChatMessage(role="user", content=user_message)
    )

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    try:
        messages = [ChatMessage(role="system", content=SYSTEM_PROMPT)] + conversations[user_id]

        response = mistral.chat(
            model="mistral-small-latest",  # Nhanh hơn large
            messages=messages,
            max_tokens=1024,
        )

        reply = response.choices[0].message.content
        reply_html = md_to_html(reply)

        conversations[user_id].append(
            ChatMessage(role="assistant", content=reply)
        )

        # Giới hạn lịch sử 20 tin nhắn
        if len(conversations[user_id]) > 20:
            conversations[user_id] = conversations[user_id][-20:]

        await update.message.reply_html(reply_html)

    except Exception as e:
        logging.error(f"Lỗi: {e}")
        await update.message.reply_html("⚠️ Có lỗi xảy ra, vui lòng thử lại.")

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(CommandHandler("model", model_info))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))
    print("🤖 Bot đang chạy...")
    app.run_polling()

if __name__ == "__main__":
    main()
