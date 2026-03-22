"""
discord_bot.py - Ultra Bolt Discord Bot (mirrors Telegram bot 100%)
Run: python discord_bot.py
"""
import os
import re
import html
import time
import asyncio
import base64
import logging
import tempfile
from collections import defaultdict
from io import BytesIO
from typing import Optional

import discord
from discord import app_commands
from discord.ext import tasks
from dotenv import load_dotenv

from database import (
    init_db, clear_history, get_setting, set_setting,
    get_profile, set_profile, get_history,
    add_allowed_user, remove_allowed_user, list_allowed_users,
)
from llm_core import (
    call_llm_stream, call_vision_stream, call_ocr_mistral, transcribe_audio,
)
from prompts import MODEL_REGISTRY
from tracker_core import get_usage_report
from money_tracker import handle_money_command
from reminder_system import set_reminder_from_text, list_reminders_text, delete_reminder
from agents_workflow import run_multi_agent_workflow, run_pro_workflow, run_agentic_loop, run_coder_workflow
from rag_core import has_docs, add_document, build_rag_context, list_docs, delete_doc
from focus_tracker import (
    add_task, get_tasks, complete_task, delete_task, clear_done_tasks,
    build_task_list, get_daily_summary, get_motivation,
    start_pomodoro, get_pomodoro_count_today,
)

load_dotenv()
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("discord_bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise RuntimeError("Thiếu DISCORD_TOKEN trong file .env")

_OWNER_ID_STR = os.getenv("DISCORD_OWNER_ID", os.getenv("OWNER_ID", ""))
OWNER_ID: int = int(_OWNER_ID_STR) if _OWNER_ID_STR.strip().isdigit() else 0

DISCORD_MSG_LIMIT = 1900   # safe buffer under 2000
MAX_MSG_LENGTH = 4000
RATE_LIMIT = 10
RATE_WINDOW = 60

# ── Per-user state (replaces PTB context.user_data) ───────────────────────────
_state: dict[int, dict] = defaultdict(dict)
_rate_tracker: dict[int, list[float]] = defaultdict(list)


# ── Helper: HTML → Discord markdown ───────────────────────────────────────────

def _fmt(text: str) -> str:
    """Convert Telegram-style HTML output to Discord markdown."""
    # Remove thinking blocks
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE).strip()
    # Preserve pre/code blocks first
    code_blocks: list[str] = []
    inline_codes: list[str] = []

    def save_pre(m):
        code_blocks.append(m.group(1))
        return f"%%CB{len(code_blocks)-1}%%"

    def save_code(m):
        inline_codes.append(m.group(1))
        return f"%%IC{len(inline_codes)-1}%%"

    text = re.sub(r'<pre>([\s\S]+?)</pre>', save_pre, text, flags=re.IGNORECASE)
    text = re.sub(r'<code>([\s\S]+?)</code>', save_code, text, flags=re.IGNORECASE)

    # HTML tags → discord markdown
    text = re.sub(r'<b>([\s\S]+?)</b>', r'**\1**', text, flags=re.IGNORECASE)
    text = re.sub(r'<i>([\s\S]+?)</i>', r'*\1*', text, flags=re.IGNORECASE)
    text = re.sub(r'<u>([\s\S]+?)</u>', r'__\1__', text, flags=re.IGNORECASE)
    text = re.sub(r'<s>([\s\S]+?)</s>', r'~~\1~~', text, flags=re.IGNORECASE)
    text = re.sub(r'<blockquote>([\s\S]+?)</blockquote>', r'> \1', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)  # strip remaining tags
    text = html.unescape(text)

    # Restore code
    for i, code in enumerate(inline_codes):
        text = text.replace(f"%%IC{i}%%", f"`{code}`")
    for i, code in enumerate(code_blocks):
        text = text.replace(f"%%CB{i}%%", f"```\n{code}\n```")

    return text


def _split_msg(text: str, limit: int = DISCORD_MSG_LIMIT) -> list[str]:
    """Split text into chunks that respect Discord's message limit."""
    if len(text) <= limit:
        return [text]
    parts = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        split_at = text.rfind('\n', 0, limit)
        if split_at <= 0:
            split_at = limit
        parts.append(text[:split_at])
        text = text[split_at:].lstrip('\n')
    return parts


def _is_rate_limited(user_id: int) -> tuple[bool, int]:
    now = time.time()
    ts = _rate_tracker[user_id]
    _rate_tracker[user_id] = [t for t in ts if now - t < RATE_WINDOW]
    if len(_rate_tracker[user_id]) >= RATE_LIMIT:
        wait = int(RATE_WINDOW - (now - _rate_tracker[user_id][0])) + 1
        return True, wait
    _rate_tracker[user_id].append(now)
    return False, 0


async def _is_authorized(user_id: int) -> bool:
    if OWNER_ID == 0:
        return True
    if user_id == OWNER_ID:
        return True
    from database import is_user_allowed
    return await is_user_allowed(user_id)


# ── Bot setup ─────────────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = discord.Client(intents=intents)
tree = app_commands.CommandGroup if False else app_commands.CommandTree(bot)


# ── UI Views ──────────────────────────────────────────────────────────────────

class RetryView(discord.ui.View):
    RETRY_MODELS = [
        ("small",      "🔹 Mistral S"),
        ("large",      "🔵 Mistral L"),
        ("qwen3",      "🌟 Qwen3"),
        ("gpt_120b",   "🧠 GPT 120B"),
        ("groq_large", "🦙 Llama 70B"),
    ]

    def __init__(self, user_id: int, user_message: str, current_key: str):
        super().__init__(timeout=120)
        self.user_id = user_id
        self.user_message = user_message
        for key, name in self.RETRY_MODELS:
            if key != current_key:
                self.add_item(RetryButton(user_id, user_message, key, name))


class RetryButton(discord.ui.Button):
    def __init__(self, user_id: int, user_message: str, model_key: str, label: str):
        super().__init__(label=label, style=discord.ButtonStyle.secondary)
        self.user_id = user_id
        self.user_message = user_message
        self.model_key = model_key

    async def callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Không phải tin nhắn của bạn.", ephemeral=True)
            return
        await interaction.response.defer()
        await _stream_reply(interaction.channel, self.user_id, self.user_message, model_key=self.model_key, save_history=False)


class VisionChoiceView(discord.ui.View):
    def __init__(self, user_id: int):
        super().__init__(timeout=120)
        self.user_id = user_id

    @discord.ui.button(label="🔤 OCR - Trích xuất chữ", style=discord.ButtonStyle.primary)
    async def ocr_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Không phải ảnh của bạn.", ephemeral=True)
            return
        await interaction.response.defer()
        _state[self.user_id]["vision_mode"] = "ocr"
        image_hq = _state[self.user_id].get("vision_image_hq") or _state[self.user_id].get("vision_image")
        if not image_hq:
            await interaction.followup.send("❌ Không tìm thấy ảnh.")
            return
        msg = await interaction.followup.send("⏳ Đang trích xuất chữ...")
        try:
            result = await call_ocr_mistral(self.user_id, image_hq)
            text = _fmt(result)
            parts = _split_msg(f"🔤 **Kết quả OCR:**\n\n{text}")
            view = OcrFollowupView(self.user_id)
            await msg.edit(content=parts[0], view=view if len(parts) == 1 else None)
            for i, part in enumerate(parts[1:], 1):
                await interaction.followup.send(part, view=view if i == len(parts)-1 else None)
        except Exception as e:
            await msg.edit(content=f"❌ OCR thất bại: {e}")

    @discord.ui.button(label="🖼 Mô tả ảnh", style=discord.ButtonStyle.success)
    async def describe_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Không phải ảnh của bạn.", ephemeral=True)
            return
        await interaction.response.defer()
        _state[self.user_id]["vision_mode"] = "describe"
        await _do_vision_describe(interaction, self.user_id, "Mô tả chi tiết ảnh này.")


class OcrFollowupView(discord.ui.View):
    def __init__(self, user_id: int):
        super().__init__(timeout=300)
        self.user_id = user_id

    @discord.ui.button(label="🖼 Mô tả ảnh", style=discord.ButtonStyle.success)
    async def describe_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Không phải ảnh của bạn.", ephemeral=True)
            return
        await interaction.response.defer()
        _state[self.user_id]["vision_mode"] = "describe"
        await _do_vision_describe(interaction, self.user_id, "Mô tả chi tiết ảnh này.")

    @discord.ui.button(label="❌ Xóa ảnh", style=discord.ButtonStyle.danger)
    async def clear_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Không phải ảnh của bạn.", ephemeral=True)
            return
        for key in ("vision_image", "vision_image_hq", "vision_caption", "vision_messages", "vision_model", "vision_mode"):
            _state[self.user_id].pop(key, None)
        await interaction.response.send_message("🗑 Đã xóa ảnh.", ephemeral=True)
        self.stop()


class VisionFollowupView(discord.ui.View):
    MODELS = [
        ("llama4",     "👁 Vision"),
        ("groq_large", "🦙 Llama 70B"),
        ("gpt_120b",   "🧠 GPT 120B"),
        ("qwen3",      "🌟 Qwen3"),
        ("kimi",       "🌙 Kimi K2"),
    ]

    def __init__(self, user_id: int, current_key: str = "llama4"):
        super().__init__(timeout=300)
        self.user_id = user_id
        for key, name in self.MODELS:
            if key != current_key:
                self.add_item(VisionModelButton(user_id, key, name))
        # Add clear button
        clear_btn = discord.ui.Button(label="❌ Xóa ảnh", style=discord.ButtonStyle.danger)
        async def _clear(interaction: discord.Interaction):
            if interaction.user.id != user_id:
                await interaction.response.send_message("Không phải ảnh của bạn.", ephemeral=True)
                return
            for key in ("vision_image", "vision_image_hq", "vision_caption", "vision_messages", "vision_model", "vision_mode"):
                _state[user_id].pop(key, None)
            await interaction.response.send_message("🗑 Đã xóa ảnh.", ephemeral=True)
        clear_btn.callback = _clear
        self.add_item(clear_btn)


class VisionModelButton(discord.ui.Button):
    def __init__(self, user_id: int, model_key: str, label: str):
        super().__init__(label=label, style=discord.ButtonStyle.secondary)
        self.user_id = user_id
        self.model_key = model_key

    async def callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Không phải ảnh của bạn.", ephemeral=True)
            return
        await interaction.response.defer()
        _state[self.user_id]["vision_model"] = self.model_key
        await _do_vision_describe(interaction, self.user_id, "Mô tả chi tiết ảnh này.", model_key=self.model_key)


class ModelSelectView(discord.ui.View):
    def __init__(self, user_id: int, current_key: str):
        super().__init__(timeout=60)
        self.user_id = user_id
        self.add_item(ModelSelectMenu(user_id, current_key))


class ModelSelectMenu(discord.ui.Select):
    def __init__(self, user_id: int, current_key: str):
        options = []
        for key, info in MODEL_REGISTRY.items():
            desc = info['desc'][:100]
            options.append(discord.SelectOption(
                label=info['name'][:100],
                value=key,
                description=desc,
                default=(key == current_key),
            ))
        super().__init__(placeholder="Chọn model AI...", options=options[:25])
        self.user_id = user_id

    async def callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Không phải yêu cầu của bạn.", ephemeral=True)
            return
        key = self.values[0]
        await set_setting(self.user_id, "model_key", key)
        await set_setting(self.user_id, "auto_mode", "0")
        name = MODEL_REGISTRY[key]['name']
        await interaction.response.send_message(f"✅ Đã chọn model: **{name}**\nAuto-routing đã tắt. Dùng `/auto` để bật lại.", ephemeral=True)


# ── Vision helper ─────────────────────────────────────────────────────────────

async def _do_vision_describe(
    interaction_or_channel,
    user_id: int,
    prompt: str,
    model_key: str = None,
):
    """Run vision describe on stored image, streaming to channel."""
    is_interaction = isinstance(interaction_or_channel, discord.Interaction)
    channel = interaction_or_channel.channel if is_interaction else interaction_or_channel

    image_b64 = _state[user_id].get("vision_image")
    caption = _state[user_id].get("vision_caption", "")
    vision_msgs = _state[user_id].get("vision_messages")
    current_model = model_key or _state[user_id].get("vision_model", "llama4")

    if not image_b64:
        msg = "❌ Không tìm thấy ảnh."
        if is_interaction:
            await interaction_or_channel.followup.send(msg)
        else:
            await channel.send(msg)
        return

    # Build or extend vision messages
    if vision_msgs is None:
        full_prompt = prompt
        if caption:
            full_prompt = f"{caption}\n\n{prompt}" if prompt != caption else caption
        vision_msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": full_prompt},
            ],
        }]
    else:
        vision_msgs.append({"role": "user", "content": prompt})

    lang_mode = await get_setting(user_id, "lang_mode", "vi")
    if lang_mode == "zh-TW":
        vision_msgs = [{
            "role": "system", "content": "請用繁體中文回答。"
        }] + vision_msgs
    else:
        vision_msgs = [{
            "role": "system", "content": "Hãy trả lời bằng tiếng Việt."
        }] + vision_msgs

    # Send placeholder
    if is_interaction:
        placeholder = await interaction_or_channel.followup.send("⏳ Đang phân tích ảnh...")
    else:
        placeholder = await channel.send("⏳ Đang phân tích ảnh...")

    full_reply = ""
    last_edit = 0.0
    try:
        async for chunk, mk in call_vision_stream(user_id, vision_msgs):
            full_reply += chunk
            now = time.monotonic()
            if now - last_edit > 0.8 and len(full_reply) < DISCORD_MSG_LIMIT:
                try:
                    await placeholder.edit(content=_fmt(full_reply) or "⏳")
                    last_edit = now
                except Exception:
                    pass
    except Exception as e:
        logger.error(f"Vision stream error: {e}")
        await placeholder.edit(content=f"❌ Lỗi xử lý ảnh: {e}")
        return

    if not full_reply:
        await placeholder.edit(content="❌ Không nhận được phản hồi.")
        return

    # Update stored vision_messages (without system prefix)
    clean_msgs = [m for m in vision_msgs if m.get("role") != "system"]
    clean_msgs.append({"role": "assistant", "content": full_reply})
    _state[user_id]["vision_messages"] = clean_msgs
    _state[user_id]["vision_model"] = current_model
    _state[user_id]["vision_mode"] = "describe"

    formatted = _fmt(full_reply)
    parts = _split_msg(formatted)
    view = VisionFollowupView(user_id, current_model)
    await placeholder.edit(content=parts[0], view=view if len(parts) == 1 else None)
    for i, part in enumerate(parts[1:], 1):
        await channel.send(part, view=view if i == len(parts)-1 else None)


# ── Streaming LLM reply helper ────────────────────────────────────────────────

async def _stream_reply(
    channel,
    user_id: int,
    user_message: str,
    model_key: str = None,
    save_history: bool = True,
    extra_context: str = None,
):
    """Stream LLM reply to Discord channel with live editing."""
    placeholder = await channel.send("⏳")
    full_reply = ""
    last_edit = 0.0
    used_key = model_key

    try:
        async for chunk, mk in call_llm_stream(
            user_id, user_message,
            model_key=model_key,
            extra_context=extra_context,
            save_history=save_history,
        ):
            full_reply += chunk
            used_key = mk
            now = time.monotonic()
            if now - last_edit > 0.8 and len(full_reply) < DISCORD_MSG_LIMIT:
                try:
                    await placeholder.edit(content=_fmt(full_reply) or "⏳")
                    last_edit = now
                except Exception:
                    pass
    except Exception as e:
        logger.error(f"LLM stream error for user {user_id}: {e}")
        await placeholder.edit(content=f"❌ Lỗi: {e}")
        return

    if not full_reply:
        await placeholder.edit(content="❌ Không nhận được phản hồi.")
        return

    formatted = _fmt(full_reply)
    model_name = MODEL_REGISTRY.get(used_key, {}).get("name", used_key)
    footer = f"\n*— {model_name}*"
    parts = _split_msg(formatted)

    view = RetryView(user_id, user_message, used_key)
    last_part = parts[-1] + (footer if len(parts[-1]) + len(footer) <= DISCORD_MSG_LIMIT else "")

    await placeholder.edit(content=parts[0] if len(parts) == 1 else parts[0])
    if len(parts) == 1:
        await placeholder.edit(content=last_part, view=view)
    else:
        for i, part in enumerate(parts[1:], 1):
            is_last = i == len(parts)-1
            await channel.send(last_part if is_last else part, view=view if is_last else None)


# ── on_message: text, photo, voice, document ─────────────────────────────────

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if not message.guild and False:  # DMs allowed
        return

    user_id = message.author.id

    if not await _is_authorized(user_id):
        await message.reply("⛔ Bot này là riêng tư.")
        return

    # ── Handle image attachments ─────────────────────────────────────────────
    image_atts = [a for a in message.attachments if a.content_type and a.content_type.startswith("image/")]
    audio_atts = [a for a in message.attachments if a.content_type and (
        a.content_type.startswith("audio/") or a.filename.lower().endswith((".ogg", ".mp3", ".wav", ".m4a", ".webm"))
    )]
    doc_atts = [a for a in message.attachments if a.content_type and (
        "pdf" in a.content_type or "word" in a.content_type or "text/plain" in a.content_type
        or a.filename.lower().endswith((".pdf", ".txt", ".docx", ".doc"))
    )]

    if image_atts:
        att = image_atts[0]
        data = await att.read()
        try:
            from PIL import Image
            img = Image.open(BytesIO(data))
            # 512px for vision
            img_512 = img.copy()
            img_512.thumbnail((512, 512), Image.LANCZOS)
            buf = BytesIO()
            img_512.save(buf, format="JPEG", quality=85)
            b64_512 = base64.b64encode(buf.getvalue()).decode()
            # 1024px HQ for OCR
            img_hq = img.copy()
            img_hq.thumbnail((1024, 1024), Image.LANCZOS)
            buf_hq = BytesIO()
            img_hq.save(buf_hq, format="JPEG", quality=90)
            b64_hq = base64.b64encode(buf_hq.getvalue()).decode()
        except ImportError:
            b64_512 = base64.b64encode(data).decode()
            b64_hq = b64_512

        _state[user_id]["vision_image"] = b64_512
        _state[user_id]["vision_image_hq"] = b64_hq
        _state[user_id]["vision_caption"] = message.content or ""
        _state[user_id]["vision_messages"] = None
        _state[user_id]["vision_mode"] = None

        view = VisionChoiceView(user_id)
        await message.reply("📷 Ảnh đã nhận. Bạn muốn làm gì?", view=view)
        return

    if audio_atts:
        att = audio_atts[0]
        data = await att.read()
        lang_mode = await get_setting(user_id, "lang_mode", "vi")
        whisper_lang = "zh" if lang_mode == "zh-TW" else "vi"

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
            f.write(data)
            tmp_path = f.name

        proc_msg = await message.reply("🎙 Đang nhận dạng giọng nói...")
        try:
            transcript = await transcribe_audio(tmp_path, language=whisper_lang)
        except Exception as e:
            await proc_msg.edit(content=f"❌ STT thất bại: {e}")
            return
        finally:
            os.unlink(tmp_path)

        if not transcript.strip():
            await proc_msg.edit(content="❌ Không nhận dạng được giọng nói.")
            return

        await proc_msg.edit(content=f"🎙 *{transcript}*\n\n⏳ Đang xử lý...")

        limited, wait = _is_rate_limited(user_id)
        if limited:
            await proc_msg.edit(content=f"⏳ Bạn gửi quá nhanh! Chờ **{wait} giây**.")
            return

        rag_ctx = None
        if await has_docs(user_id):
            rag_ctx = await build_rag_context(user_id, transcript)

        full_reply = ""
        used_key = None
        try:
            async for chunk, mk in call_llm_stream(user_id, transcript, extra_context=rag_ctx):
                full_reply += chunk
                used_key = mk
        except Exception as e:
            await proc_msg.edit(content=f"❌ Lỗi: {e}")
            return

        formatted = _fmt(full_reply)
        model_name = MODEL_REGISTRY.get(used_key, {}).get("name", used_key)
        parts = _split_msg(f"🎙 *{transcript}*\n\n{formatted}\n\n*— {model_name}*")
        view = RetryView(user_id, transcript, used_key)
        await proc_msg.edit(content=parts[0], view=view if len(parts) == 1 else None)
        for i, part in enumerate(parts[1:], 1):
            await message.channel.send(part, view=view if i == len(parts)-1 else None)
        return

    if doc_atts:
        att = doc_atts[0]
        data = await att.read()
        proc_msg = await message.reply(f"📄 Đang xử lý tài liệu **{att.filename}**...")
        try:
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(att.filename)[1], delete=False) as f:
                f.write(data)
                tmp_path = f.name
            result = await add_document(user_id, tmp_path, att.filename)
            await proc_msg.edit(content=result)
        except Exception as e:
            await proc_msg.edit(content=f"❌ Lỗi xử lý tài liệu: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        return

    # ── Text message ─────────────────────────────────────────────────────────
    text = message.content.strip()
    if not text:
        return

    # Vision follow-up
    vision_mode = _state[user_id].get("vision_mode")
    if vision_mode == "describe" and _state[user_id].get("vision_image"):
        limited, wait = _is_rate_limited(user_id)
        if limited:
            await message.reply(f"⏳ Bạn gửi quá nhanh! Chờ **{wait} giây**.")
            return
        await _do_vision_describe(message.channel, user_id, text)
        return
    elif vision_mode == "ocr" and _state[user_id].get("vision_image"):
        # OCR mode follow-up: describe the image
        _state[user_id]["vision_mode"] = "describe"
        limited, wait = _is_rate_limited(user_id)
        if limited:
            await message.reply(f"⏳ Bạn gửi quá nhanh! Chờ **{wait} giây**.")
            return
        await _do_vision_describe(message.channel, user_id, text)
        return

    if len(text) > MAX_MSG_LENGTH:
        await message.reply(f"⚠️ Tin nhắn quá dài (**{len(text)}/{MAX_MSG_LENGTH}** ký tự).")
        return

    limited, wait = _is_rate_limited(user_id)
    if limited:
        await message.reply(f"⏳ Bạn gửi quá nhanh! Chờ **{wait} giây**.")
        return

    # RAG context
    rag_ctx = None
    if await has_docs(user_id):
        rag_ctx = await build_rag_context(user_id, text)

    async with message.channel.typing():
        pass

    await _stream_reply(message.channel, user_id, text, extra_context=rag_ctx)


# ── Slash commands ────────────────────────────────────────────────────────────

@tree.command(name="start", description="Bắt đầu / giới thiệu bot")
async def cmd_start(interaction: discord.Interaction):
    name = interaction.user.display_name
    await interaction.response.send_message(
        f"⚡ **Ultra Bolt** đã sẵn sàng!\n\n"
        f"Xin chào, **{name}**! Tôi là trợ lý AI powered by Mistral AI.\n\n"
        f"🧠 **Tính năng:**\n"
        f"• Chat thông minh với auto-routing model\n"
        f"• Xử lý ảnh & giọng nói\n"
        f"• Đặt nhắc nhở bằng ngôn ngữ tự nhiên\n"
        f"• Theo dõi chi tiêu cá nhân\n"
        f"• Upload tài liệu & hỏi nội dung (RAG)\n"
        f"• Multi-agent AI workflows\n\n"
        f"Gõ `/help` để xem tất cả lệnh."
    )


@tree.command(name="help", description="Danh sách tất cả lệnh")
async def cmd_help(interaction: discord.Interaction):
    await interaction.response.send_message(
        "📖 **Danh sách lệnh đầy đủ:**\n\n"
        "**💬 Chat:**\n"
        "`/clear` — Xóa lịch sử hội thoại\n"
        "`/model` — Chọn model AI\n"
        "`/models` — Danh sách tất cả models\n"
        "`/auto` — Bật/tắt tự động chọn model\n"
        "`/profile [text]` — Xem/đặt hồ sơ cá nhân\n"
        "`/stats` — Thống kê phiên chat\n\n"
        "**🌐 Ngôn ngữ:**\n"
        "`/tw` — Chuyển sang tiếng Trung phồn thể\n"
        "`/vi` — Chuyển về tiếng Việt\n\n"
        "**🤖 AI Nâng cao:**\n"
        "`/pro <task>` — Phân tích sâu với Mistral Large\n"
        "`/agent <task>` — Agentic loop tự động (5 bước)\n"
        "`/coder <task>` — Workflow lập trình chuyên sâu\n\n"
        "**⏰ Nhắc nhở:**\n"
        "`/remind <text>` — Đặt nhắc nhở (hỗ trợ tiếng Việt)\n"
        "`/reminders` — Danh sách nhắc nhở\n\n"
        "**💰 Tài chính:**\n"
        "`/mn +500 cà phê` — Thêm thu nhập\n"
        "`/mn -200 ăn trưa` — Thêm chi tiêu\n"
        "`/mn` — Thống kê tháng này\n\n"
        "**📚 RAG (Tài liệu):**\n"
        "`/rag list` — Danh sách tài liệu\n"
        "`/rag clear <tên file>` — Xóa tài liệu\n"
        "Gửi file PDF/TXT/DOCX để thêm vào RAG\n\n"
        "**📋 Công việc:**\n"
        "`/todo <task>` — Thêm việc cần làm\n"
        "`/tasks` — Danh sách công việc\n"
        "`/done <id>` — Đánh dấu hoàn thành\n"
        "`/deltask <id>` — Xóa task\n"
        "`/pomodoro` — Bắt đầu Pomodoro\n"
        "`/motivation` — Động lực hằng ngày\n"
        "`/checkin` — Báo cáo ngày\n\n"
        "**📊 Thống kê:**\n"
        "`/tokens` — Chi phí và token đã dùng",
        ephemeral=True,
    )


@tree.command(name="clear", description="Xóa lịch sử hội thoại")
async def cmd_clear(interaction: discord.Interaction):
    user_id = interaction.user.id
    await clear_history(user_id)
    for key in ("vision_image", "vision_image_hq", "vision_caption", "vision_messages", "vision_model", "vision_mode"):
        _state[user_id].pop(key, None)
    await interaction.response.send_message("🗑 Đã xóa toàn bộ lịch sử hội thoại.", ephemeral=True)


@tree.command(name="model", description="Chọn model AI")
async def cmd_model(interaction: discord.Interaction):
    user_id = interaction.user.id
    current = await get_setting(user_id, "model_key", "groq_large")
    view = ModelSelectView(user_id, current)
    current_name = MODEL_REGISTRY.get(current, {}).get("name", current)
    await interaction.response.send_message(
        f"🤖 Model hiện tại: **{current_name}**\n\nChọn model:",
        view=view,
        ephemeral=True,
    )


@tree.command(name="models", description="Danh sách tất cả models")
async def cmd_models(interaction: discord.Interaction):
    user_id = interaction.user.id
    current = await get_setting(user_id, "model_key", "groq_large")
    lines = ["🤖 **Danh sách models:**\n"]
    for key, info in MODEL_REGISTRY.items():
        active = " ✓" if key == current else ""
        lines.append(f"{info['name']}{active}")
        lines.append(f"_{info['desc']}_")
        lines.append(f"`{info['model_id']}`\n")
    await interaction.response.send_message("\n".join(lines), ephemeral=True)


@tree.command(name="auto", description="Bật/tắt tự động chọn model")
async def cmd_auto(interaction: discord.Interaction):
    user_id = interaction.user.id
    current = await get_setting(user_id, "auto_mode", "1")
    new_val = "0" if current == "1" else "1"
    await set_setting(user_id, "auto_mode", new_val)
    if new_val == "1":
        await interaction.response.send_message(
            "🔄 **Tự động chọn model: BẬT**\n\nBot sẽ tự chọn model phù hợp nhất.", ephemeral=True
        )
    else:
        model_key = await get_setting(user_id, "model_key", "groq_large")
        model_name = MODEL_REGISTRY.get(model_key, {}).get("name", model_key)
        await interaction.response.send_message(
            f"🔒 **Tự động chọn model: TẮT**\n\nSẽ dùng cố định: **{model_name}**\nDùng `/model` để thay đổi.", ephemeral=True
        )


@tree.command(name="tw", description="Chuyển sang tiếng Trung phồn thể")
async def cmd_tw(interaction: discord.Interaction):
    user_id = interaction.user.id
    await set_setting(user_id, "lang_mode", "zh-TW")
    await interaction.response.send_message(
        "🇹🇼 已切換到繁體中文模式。\n(Đã chuyển sang chế độ tiếng Trung phồn thể)", ephemeral=True
    )


@tree.command(name="vi", description="Chuyển về tiếng Việt")
async def cmd_vi(interaction: discord.Interaction):
    user_id = interaction.user.id
    await set_setting(user_id, "lang_mode", "vi")
    await interaction.response.send_message(
        "🇻🇳 Đã chuyển về chế độ tiếng Việt.", ephemeral=True
    )


@tree.command(name="profile", description="Xem/đặt hồ sơ cá nhân")
@app_commands.describe(text="Nội dung hồ sơ (bỏ trống để xem)")
async def cmd_profile(interaction: discord.Interaction, text: str = ""):
    user_id = interaction.user.id
    if text:
        await set_profile(user_id, text)
        await interaction.response.send_message(f"✅ Đã cập nhật hồ sơ:\n_{text}_", ephemeral=True)
    else:
        profile = await get_profile(user_id)
        if profile:
            await interaction.response.send_message(f"👤 **Hồ sơ của bạn:**\n_{profile}_", ephemeral=True)
        else:
            await interaction.response.send_message(
                "👤 Bạn chưa có hồ sơ.\nDùng `/profile [mô tả]` để thêm.", ephemeral=True
            )


@tree.command(name="stats", description="Thống kê phiên chat")
async def cmd_stats(interaction: discord.Interaction):
    user_id = interaction.user.id
    history = await get_history(user_id, limit=100)
    model_key = await get_setting(user_id, "model_key", "groq_large")
    auto_mode = await get_setting(user_id, "auto_mode", "1")
    user_msgs = sum(1 for m in history if m["role"] == "user")
    model_name = MODEL_REGISTRY.get(model_key, {}).get("name", model_key)
    auto_str = "BẬT" if auto_mode == "1" else "TẮT"
    await interaction.response.send_message(
        f"📊 **Thống kê của bạn**\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💬 Tin nhắn:  **{user_msgs}**\n"
        f"🤖 Model:     **{model_name}**\n"
        f"🔄 Auto:      **{auto_str}**\n"
        f"━━━━━━━━━━━━━━━━━━━━",
        ephemeral=True,
    )


@tree.command(name="tokens", description="Chi phí và token đã dùng")
async def cmd_tokens(interaction: discord.Interaction):
    user_id = interaction.user.id
    report = await get_usage_report(user_id)
    await interaction.response.send_message(_fmt(report), ephemeral=True)


@tree.command(name="remind", description="Đặt nhắc nhở bằng ngôn ngữ tự nhiên")
@app_commands.describe(text="Nội dung và thời gian nhắc nhở")
async def cmd_remind(interaction: discord.Interaction, text: str = ""):
    user_id = interaction.user.id
    if not text:
        await interaction.response.send_message(
            "⏰ **Đặt nhắc nhở:**\n\n"
            "Cú pháp: `/remind [nội dung] [thời gian]`\n\n"
            "Ví dụ:\n"
            "`/remind họp team lúc 9h sáng ngày mai`\n"
            "`/remind uống thuốc 7h tối`\n"
            "`/remind gửi báo cáo sau 30 phút`",
            ephemeral=True,
        )
        return
    await interaction.response.defer(ephemeral=True)
    result = await set_reminder_from_text(user_id, text)
    await interaction.followup.send(_fmt(result))


@tree.command(name="reminders", description="Danh sách nhắc nhở")
async def cmd_reminders(interaction: discord.Interaction):
    user_id = interaction.user.id
    text = await list_reminders_text(user_id)
    await interaction.response.send_message(_fmt(text), ephemeral=True)


@tree.command(name="mn", description="Quản lý chi tiêu cá nhân")
@app_commands.describe(args="Ví dụ: +500 cà phê | -200 ăn trưa | week | all | del <id>")
async def cmd_mn(interaction: discord.Interaction, args: str = ""):
    user_id = interaction.user.id
    await interaction.response.defer(ephemeral=True)
    result = await handle_money_command(user_id, args)
    await interaction.followup.send(_fmt(result))


@tree.command(name="pro", description="Phân tích sâu với Mistral Large")
@app_commands.describe(task="Câu hỏi hoặc nhiệm vụ cần phân tích")
async def cmd_pro(interaction: discord.Interaction, task: str = ""):
    user_id = interaction.user.id
    if not task:
        await interaction.response.send_message(
            "🔬 **Pro Workflow**\nDùng `/pro [câu hỏi]` để phân tích sâu.", ephemeral=True
        )
        return
    await interaction.response.defer()
    result = await run_pro_workflow(user_id, task)
    parts = _split_msg(_fmt(result))
    await interaction.followup.send(parts[0])
    for part in parts[1:]:
        await interaction.channel.send(part)


@tree.command(name="agent", description="Agentic loop tự động (5 bước)")
@app_commands.describe(task="Nhiệm vụ cần thực hiện")
async def cmd_agent(interaction: discord.Interaction, task: str = ""):
    user_id = interaction.user.id
    if not task:
        await interaction.response.send_message(
            "🤖 **Agentic Loop**\nDùng `/agent [nhiệm vụ]` để bắt đầu.", ephemeral=True
        )
        return
    await interaction.response.defer()
    result = await run_agentic_loop(user_id, task)
    parts = _split_msg(_fmt(result))
    await interaction.followup.send(parts[0])
    for part in parts[1:]:
        await interaction.channel.send(part)


@tree.command(name="coder", description="Workflow lập trình chuyên sâu")
@app_commands.describe(task="Yêu cầu lập trình")
async def cmd_coder(interaction: discord.Interaction, task: str = ""):
    user_id = interaction.user.id
    if not task:
        await interaction.response.send_message(
            "💻 **Coder Workflow**\nDùng `/coder [yêu cầu]` để bắt đầu.", ephemeral=True
        )
        return
    await interaction.response.defer()
    result = await run_coder_workflow(user_id, task)
    parts = _split_msg(_fmt(result))
    await interaction.followup.send(parts[0])
    for part in parts[1:]:
        await interaction.channel.send(part)


@tree.command(name="rag", description="Quản lý tài liệu RAG")
@app_commands.describe(action="list | clear", filename="Tên file cần xóa (dùng với clear)")
async def cmd_rag(interaction: discord.Interaction, action: str = "list", filename: str = ""):
    user_id = interaction.user.id
    if action.lower() == "list":
        text = await list_docs(user_id)
        await interaction.response.send_message(_fmt(text), ephemeral=True)
    elif action.lower() == "clear" and filename:
        result = await delete_doc(user_id, filename)
        await interaction.response.send_message(_fmt(result), ephemeral=True)
    else:
        await interaction.response.send_message(
            "📚 **RAG Commands:**\n`/rag list` — Danh sách\n`/rag clear <filename>` — Xóa",
            ephemeral=True,
        )


@tree.command(name="todo", description="Thêm việc cần làm")
@app_commands.describe(task="Nội dung task")
async def cmd_todo(interaction: discord.Interaction, task: str = ""):
    user_id = interaction.user.id
    if not task:
        await interaction.response.send_message("Dùng `/todo [nội dung task]`", ephemeral=True)
        return
    result = await add_task(user_id, task)
    await interaction.response.send_message(_fmt(result), ephemeral=True)


@tree.command(name="tasks", description="Danh sách công việc")
async def cmd_tasks(interaction: discord.Interaction):
    user_id = interaction.user.id
    text = await build_task_list(user_id)
    await interaction.response.send_message(_fmt(text), ephemeral=True)


@tree.command(name="done", description="Đánh dấu task hoàn thành")
@app_commands.describe(task_id="ID của task")
async def cmd_done(interaction: discord.Interaction, task_id: int):
    user_id = interaction.user.id
    result = await complete_task(user_id, task_id)
    await interaction.response.send_message(_fmt(result), ephemeral=True)


@tree.command(name="deltask", description="Xóa task")
@app_commands.describe(task_id="ID của task")
async def cmd_deltask(interaction: discord.Interaction, task_id: int):
    user_id = interaction.user.id
    result = await delete_task(user_id, task_id)
    await interaction.response.send_message(_fmt(result), ephemeral=True)


@tree.command(name="pomodoro", description="Bắt đầu Pomodoro 25 phút")
async def cmd_pomodoro(interaction: discord.Interaction):
    user_id = interaction.user.id
    result = await start_pomodoro(user_id)
    await interaction.response.send_message(_fmt(result), ephemeral=True)


@tree.command(name="motivation", description="Câu động lực hằng ngày")
async def cmd_motivation(interaction: discord.Interaction):
    user_id = interaction.user.id
    result = await get_motivation(user_id)
    await interaction.response.send_message(_fmt(result))


@tree.command(name="checkin", description="Báo cáo ngày hôm nay")
async def cmd_checkin(interaction: discord.Interaction):
    user_id = interaction.user.id
    result = await get_daily_summary(user_id)
    await interaction.response.send_message(_fmt(result), ephemeral=True)


@tree.command(name="user", description="Quản lý user được phép (owner only)")
@app_commands.describe(action="add | remove | list", user_id_str="User ID (dùng với add/remove)")
async def cmd_user(interaction: discord.Interaction, action: str = "list", user_id_str: str = ""):
    caller = interaction.user.id
    if OWNER_ID != 0 and caller != OWNER_ID:
        await interaction.response.send_message("⛔ Chỉ owner mới dùng được lệnh này.", ephemeral=True)
        return
    if action == "list":
        users = await list_allowed_users()
        if not users:
            await interaction.response.send_message("📋 Chưa có user nào trong whitelist.", ephemeral=True)
        else:
            await interaction.response.send_message(
                "📋 **Danh sách user được phép:**\n" + "\n".join(f"• `{u}`" for u in users),
                ephemeral=True,
            )
    elif action == "add" and user_id_str.isdigit():
        await add_allowed_user(int(user_id_str))
        await interaction.response.send_message(f"✅ Đã thêm user `{user_id_str}`.", ephemeral=True)
    elif action == "remove" and user_id_str.isdigit():
        await remove_allowed_user(int(user_id_str))
        await interaction.response.send_message(f"✅ Đã xóa user `{user_id_str}`.", ephemeral=True)
    else:
        await interaction.response.send_message(
            "Cú pháp: `/user list` | `/user add <id>` | `/user remove <id>`", ephemeral=True
        )


# ── Reminder loop ─────────────────────────────────────────────────────────────

@tasks.loop(seconds=30)
async def reminder_check():
    """Send due reminders via Discord DM (mirrors Telegram reminder_loop logic)."""
    from database import get_pending_reminders, delete_reminder as db_delete_reminder, update_reminder_fire_at
    from datetime import datetime, timedelta, timezone

    def _now_utc():
        return datetime.now(timezone.utc).replace(tzinfo=None)

    try:
        pending = await get_pending_reminders()
    except Exception as e:
        logger.error(f"Reminder check error: {e}")
        return

    for reminder in pending:
        uid = reminder["user_id"]
        try:
            discord_user = await bot.fetch_user(uid)
            if discord_user:
                await discord_user.send(f"⏰ **Nhắc nhở!**\n\n{reminder['message']}")
            logger.info(f"Reminder #{reminder['id']} fired for Discord user {uid}")
        except Exception as e:
            logger.warning(f"Could not send reminder DM to {uid}: {e}")

        if reminder.get("repeat"):
            try:
                interval_sec = int(reminder["repeat"])
                current_fire = datetime.fromisoformat(reminder["fire_at"])
                next_fire = current_fire + timedelta(seconds=interval_sec)
                now = _now_utc()
                while next_fire <= now:
                    next_fire += timedelta(seconds=interval_sec)
                await update_reminder_fire_at(reminder["id"], next_fire.isoformat())
            except Exception as e:
                logger.error(f"Failed to reschedule reminder #{reminder['id']}: {e}")
                await db_delete_reminder(reminder["id"])
        else:
            await db_delete_reminder(reminder["id"])


# ── Bot events ────────────────────────────────────────────────────────────────

@bot.event
async def on_ready():
    await init_db()
    logger.info(f"Discord bot logged in as {bot.user} (ID: {bot.user.id})")
    try:
        synced = await tree.sync()
        logger.info(f"Synced {len(synced)} slash commands")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")
    reminder_check.start()
    logger.info("Ultra Bolt Discord Bot is ready!")


@bot.event
async def on_error(event, *args, **kwargs):
    logger.exception(f"Discord error in event {event}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN, log_handler=None)
