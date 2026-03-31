"""
discord_bot.py - Ultra Bolt Discord Bot (mirrors Telegram bot 100%)
Run: python discord_bot.py

FIXES applied vs original:
  BUG 1: on_ready + tree.sync() — slash commands were never registered with Discord
  BUG 2: add_document() arg order was wrong (tmp_path, filename) → (filename, bytes)
  BUG 3: entry point was async def main() / asyncio.run() — replaced with bot.run()
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
from api_dashboard import cmd_mapi_discord, cmd_gapi_discord, cmd_gemapi_discord
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

DISCORD_MSG_LIMIT = 1900
RATE_LIMIT = 10
RATE_WINDOW = 60

# ── Per-user state ────────────────────────────────────────────────────────────
_state: dict[int, dict] = defaultdict(dict)
_rate_tracker: dict[int, list[float]] = defaultdict(list)

# ── HTML → Discord markdown ───────────────────────────────────────────────────
def _fmt(text: str) -> str:
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE).strip()
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
    text = re.sub(r'<b>([\s\S]+?)</b>', r'**\1**', text, flags=re.IGNORECASE)
    text = re.sub(r'<i>([\s\S]+?)</i>', r'*\1*', text, flags=re.IGNORECASE)
    text = re.sub(r'<u>([\s\S]+?)</u>', r'__\1__', text, flags=re.IGNORECASE)
    text = re.sub(r'<s>([\s\S]+?)</s>', r'~~\1~~', text, flags=re.IGNORECASE)
    text = re.sub(r'<blockquote>([\s\S]+?)</blockquote>', r'> \1', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = html.unescape(text)

    for i, code in enumerate(inline_codes):
        text = text.replace(f"%%IC{i}%%", f"`{code}`")
    for i, code in enumerate(code_blocks):
        text = text.replace(f"%%CB{i}%%", f"```\n{code}\n```")
    return text


def _split_msg(text: str, limit: int = DISCORD_MSG_LIMIT) -> list[str]:
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
tree = app_commands.CommandTree(bot)


# ── UI Views ──────────────────────────────────────────────────────────────────

class RetryView(discord.ui.View):
    RETRY_MODELS = [
        ("flash", "⚡ Flash"),
        ("flash_lite", "💨 Flash Lite"),
        ("flash_think", "💭 Qwen3 32B"),
        ("pro", "🧠 GPT-OSS 120B"),
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
        await _stream_reply(interaction.channel, self.user_id, self.user_message,
                            model_key=self.model_key, save_history=False)


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
        for key in ("vision_image", "vision_image_hq", "vision_caption",
                    "vision_messages", "vision_model", "vision_mode"):
            _state[self.user_id].pop(key, None)
        await interaction.response.send_message("🗑 Đã xóa ảnh.", ephemeral=True)
        self.stop()


class VisionFollowupView(discord.ui.View):
    MODELS = [
        ("flash", "⚡ Flash"),
        ("pro", "🧠 GPT-OSS 120B"),
        ("flash_think", "💭 Qwen3 32B"),
    ]

    def __init__(self, user_id: int, current_key: str = "flash"):
        super().__init__(timeout=300)
        self.user_id = user_id
        for key, name in self.MODELS:
            if key != current_key:
                self.add_item(VisionModelButton(user_id, key, name))
        clear_btn = discord.ui.Button(label="❌ Xóa ảnh", style=discord.ButtonStyle.danger)
        async def _clear(interaction: discord.Interaction):
            if interaction.user.id != user_id:
                await interaction.response.send_message("Không phải ảnh của bạn.", ephemeral=True)
                return
            for key in ("vision_image", "vision_image_hq", "vision_caption",
                        "vision_messages", "vision_model", "vision_mode"):
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
        await _do_vision_describe(interaction, self.user_id, "Mô tả chi tiết ảnh này.",
                                  model_key=self.model_key)


class ModelSelectView(discord.ui.View):
    def __init__(self, user_id: int, current_key: str):
        super().__init__(timeout=60)
        self.add_item(ModelSelectMenu(user_id, current_key))


class ModelSelectMenu(discord.ui.Select):
    def __init__(self, user_id: int, current_key: str):
        options = [
            discord.SelectOption(
                label=info['name'][:100],
                value=key,
                description=info['desc'][:100],
                default=(key == current_key),
            )
            for key, info in MODEL_REGISTRY.items()
        ]
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
        await interaction.response.send_message(
            f"✅ Đã chọn model: **{name}**\nAuto-routing đã tắt. Dùng `/auto` để bật lại.",
            ephemeral=True,
        )


# ── Vision helper ─────────────────────────────────────────────────────────────
async def _do_vision_describe(
    interaction_or_channel,
    user_id: int,
    prompt: str,
    model_key: str = None,
):
    is_interaction = isinstance(interaction_or_channel, discord.Interaction)
    channel = interaction_or_channel.channel if is_interaction else interaction_or_channel
    image_b64 = _state[user_id].get("vision_image")
    caption = _state[user_id].get("vision_caption", "")
    vision_msgs = _state[user_id].get("vision_messages")
    current_model = model_key or _state[user_id].get("vision_model", "flash")

    if not image_b64:
        msg = "❌ Không tìm thấy ảnh."
        if is_interaction:
            await interaction_or_channel.followup.send(msg)
        else:
            await channel.send(msg)
        return

    if vision_msgs is None:
        full_prompt = caption + "\n\n" + prompt if caption and prompt != caption else prompt
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
    sys_msg = "請用繁體中文回答。" if lang_mode == "zh-TW" else "Hãy trả lời bằng tiếng Việt."
    vision_msgs_with_sys = [{"role": "system", "content": sys_msg}] + vision_msgs

    placeholder = await (interaction_or_channel.followup.send("⏳ Đang phân tích ảnh...")
                         if is_interaction else channel.send("⏳ Đang phân tích ảnh..."))

    full_reply = ""
    last_edit = 0.0
    try:
        async for chunk, mk in call_vision_stream(user_id, vision_msgs_with_sys):
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

    vision_msgs.append({"role": "assistant", "content": full_reply})
    _state[user_id]["vision_messages"] = vision_msgs
    _state[user_id]["vision_model"] = current_model
    _state[user_id]["vision_mode"] = "describe"

    formatted = _fmt(full_reply)
    parts = _split_msg(formatted)
    view = VisionFollowupView(user_id, current_model)
    await placeholder.edit(content=parts[0], view=view if len(parts) == 1 else None)
    for i, part in enumerate(parts[1:], 1):
        await channel.send(part, view=view if i == len(parts)-1 else None)


# ── Streaming LLM reply ───────────────────────────────────────────────────────
async def _stream_reply(
    channel,
    user_id: int,
    user_message: str,
    model_key: str = None,
    save_history: bool = True,
    extra_context: str = None,
):
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
        logger.error(f"LLM stream error user {user_id}: {e}")
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

    if len(parts) == 1:
        await placeholder.edit(content=last_part, view=view)
    else:
        await placeholder.edit(content=parts[0])
        for i, part in enumerate(parts[1:], 1):
            is_last = i == len(parts) - 1
            await channel.send(last_part if is_last else part, view=view if is_last else None)


# ── FIX 1: on_ready — MUST sync slash commands ────────────────────────────────
@bot.event
async def on_ready():
    await init_db()
    try:
        synced = await tree.sync()
        logger.info(f"✅ {bot.user} ready | {len(synced)} slash commands synced")
    except Exception as e:
        logger.error(f"❌ tree.sync() failed: {e}")

    # Reminder loop with Discord DM adapter
    try:
        class _DiscordAdapter:
            def __init__(self, b): self._b = b
            async def send_message(self, chat_id: int, text: str, **_):
                try:
                    u = await self._b.fetch_user(chat_id)
                    if u:
                        await u.send(re.sub(r'<[^>]+>', '', text)[:1900])
                except Exception as e:
                    logger.warning(f"Reminder DM {chat_id}: {e}")

        from reminder_system import reminder_loop
        asyncio.create_task(reminder_loop(_DiscordAdapter(bot)))
    except Exception as e:
        logger.warning(f"reminder_loop not started: {e}")


# ── on_message ────────────────────────────────────────────────────────────────
@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    user_id = message.author.id
    if not await _is_authorized(user_id):
        await message.reply("⛔ Bot này là riêng tư.")
        return

    image_atts = [a for a in message.attachments
                  if a.content_type and a.content_type.startswith("image/")]
    audio_atts = [a for a in message.attachments if a.content_type and (
        a.content_type.startswith("audio/") or
        a.filename.lower().endswith((".ogg", ".mp3", ".wav", ".m4a", ".webm"))
    )]
    doc_atts = [a for a in message.attachments if a.content_type and (
        "pdf" in a.content_type or "word" in a.content_type or "text/plain" in a.content_type
        or a.filename.lower().endswith((".pdf", ".txt", ".docx", ".doc"))
    )]

    # ── Image ──────────────────────────────────────────────────────────────
    if image_atts:
        att = image_atts[0]
        data = await att.read()
        try:
            from PIL import Image
            img = Image.open(BytesIO(data))
            img_512 = img.copy(); img_512.thumbnail((512, 512), Image.LANCZOS)
            buf = BytesIO(); img_512.save(buf, format="JPEG", quality=85)
            b64_512 = base64.b64encode(buf.getvalue()).decode()
            img_hq = img.copy(); img_hq.thumbnail((1024, 1024), Image.LANCZOS)
            buf_hq = BytesIO(); img_hq.save(buf_hq, format="JPEG", quality=90)
            b64_hq = base64.b64encode(buf_hq.getvalue()).decode()
        except ImportError:
            b64_512 = b64_hq = base64.b64encode(data).decode()

        _state[user_id].update({
            "vision_image": b64_512, "vision_image_hq": b64_hq,
            "vision_caption": message.content or "",
            "vision_messages": None, "vision_mode": None,
        })
        await message.reply("📷 Ảnh đã nhận. Bạn muốn làm gì?", view=VisionChoiceView(user_id))
        return

    # ── Audio ──────────────────────────────────────────────────────────────
    if audio_atts:
        att = audio_atts[0]
        data = await att.read()
        lang_mode = await get_setting(user_id, "lang_mode", "vi")
        whisper_lang = "zh" if lang_mode == "zh-TW" else "vi"
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
            f.write(data); tmp_path = f.name
        proc_msg = await message.reply("🎙 Đang nhận dạng giọng nói...")
        try:
            transcript = await transcribe_audio(tmp_path, language=whisper_lang)
        except Exception as e:
            await proc_msg.edit(content=f"❌ STT thất bại: {e}"); return
        finally:
            os.unlink(tmp_path)

        if not transcript.strip():
            await proc_msg.edit(content="❌ Không nhận dạng được giọng nói."); return

        limited, wait = _is_rate_limited(user_id)
        if limited:
            await proc_msg.edit(content=f"⏳ Chờ **{wait} giây**."); return

        await proc_msg.edit(content=f"🎙 *{transcript}*\n\n⏳")
        rag_ctx = await build_rag_context(user_id, transcript) if await has_docs(user_id) else None
        full_reply = ""; used_key = None
        try:
            async for chunk, mk in call_llm_stream(user_id, transcript, extra_context=rag_ctx):
                full_reply += chunk; used_key = mk
        except Exception as e:
            await proc_msg.edit(content=f"❌ Lỗi: {e}"); return

        model_name = MODEL_REGISTRY.get(used_key, {}).get("name", used_key)
        parts = _split_msg(f"🎙 *{transcript}*\n\n{_fmt(full_reply)}\n\n*— {model_name}*")
        view = RetryView(user_id, transcript, used_key)
        await proc_msg.edit(content=parts[0], view=view if len(parts) == 1 else None)
        for i, part in enumerate(parts[1:], 1):
            await message.channel.send(part, view=view if i == len(parts)-1 else None)
        return

    # ── FIX 2: Document — correct add_document(user_id, filename, bytes) ──
    if doc_atts:
        att = doc_atts[0]
        proc_msg = await message.reply(f"📄 Đang xử lý **{att.filename}**...")
        try:
            data = await att.read()
            result = await add_document(user_id, att.filename, data)
            await proc_msg.edit(content=result)
        except Exception as e:
            logger.error(f"Document error user {user_id}: {e}", exc_info=True)
            await proc_msg.edit(content=f"❌ Lỗi: {e}")
        return

    # ── Text ───────────────────────────────────────────────────────────────
    text = message.content.strip()
    if not text:
        return

    vision_mode = _state[user_id].get("vision_mode")
    if vision_mode in ("describe", "ocr") and _state[user_id].get("vision_image"):
        _state[user_id]["vision_mode"] = "describe"
        limited, wait = _is_rate_limited(user_id)
        if limited:
            await message.reply(f"⏳ Chờ **{wait} giây**."); return
        await _do_vision_describe(message.channel, user_id, text)
        return

    limited, wait = _is_rate_limited(user_id)
    if limited:
        await message.reply(f"⏳ Bạn gửi quá nhanh! Chờ **{wait} giây**."); return

    rag_ctx = await build_rag_context(user_id, text) if await has_docs(user_id) else None
    await _stream_reply(message.channel, user_id, text, extra_context=rag_ctx)


# ── Slash commands ────────────────────────────────────────────────────────────

@tree.command(name="start", description="Bắt đầu / giới thiệu bot")
async def cmd_start(interaction: discord.Interaction):
    await interaction.response.send_message(
        f"⚡ **Ultra Bolt** đã sẵn sàng!\n\n"
        f"Xin chào, **{interaction.user.display_name}**!\n\n"
        f"🧠 Chat AI · 📸 Vision · 🎙 STT · 📚 RAG · ⏰ Nhắc nhở · 💰 Chi tiêu\n\n"
        f"Gõ `/help` để xem tất cả lệnh."
    )

@tree.command(name="help", description="Danh sách tất cả lệnh")
async def cmd_help(interaction: discord.Interaction):
    await interaction.response.send_message(
        "📖 **Lệnh:**\n\n"
        "**Chat:** `/clear` `/model` `/models` `/auto` `/profile` `/stats`\n"
        "**Ngôn ngữ:** `/tw` `/vi`\n"
        "**AI nâng cao:** `/pro` `/agent` `/coder`\n"
        "**Nhắc nhở:** `/remind` `/reminders`\n"
        "**Tài chính:** `/mn`\n"
        "**RAG:** `/rag list|clear`  *(gửi PDF/TXT/DOCX để thêm)*\n"
        "**Việc làm:** `/todo` `/tasks` `/done` `/deltask` `/pomodoro` `/checkin`\n"
        "**Thống kê:** `/tokens`",
        ephemeral=True,
    )

@tree.command(name="clear", description="Xóa lịch sử hội thoại")
async def cmd_clear(interaction: discord.Interaction):
    user_id = interaction.user.id
    await clear_history(user_id)
    for k in ("vision_image","vision_image_hq","vision_caption","vision_messages","vision_model","vision_mode"):
        _state[user_id].pop(k, None)
    await interaction.response.send_message("🗑 Đã xóa lịch sử.", ephemeral=True)

@tree.command(name="model", description="Chọn model AI")
async def cmd_model(interaction: discord.Interaction):
    user_id = interaction.user.id
    current = await get_setting(user_id, "model_key", "flash")
    await interaction.response.send_message(
        f"🤖 Model hiện tại: **{MODEL_REGISTRY.get(current,{}).get('name', current)}**",
        view=ModelSelectView(user_id, current), ephemeral=True,
    )

@tree.command(name="models", description="Danh sách tất cả models")
async def cmd_models(interaction: discord.Interaction):
    current = await get_setting(interaction.user.id, "model_key", "flash")
    lines = ["🤖 **Models:**\n"]
    for key, info in MODEL_REGISTRY.items():
        lines.append(f"{'✓ ' if key == current else ''}{info['name']} — _{info['desc']}_")
    await interaction.response.send_message("\n".join(lines), ephemeral=True)

@tree.command(name="auto", description="Bật/tắt tự động chọn model")
async def cmd_auto(interaction: discord.Interaction):
    user_id = interaction.user.id
    cur = await get_setting(user_id, "auto_mode", "1")
    new = "0" if cur == "1" else "1"
    await set_setting(user_id, "auto_mode", new)
    await interaction.response.send_message(
        f"🔄 Auto-routing: **{'BẬT' if new=='1' else 'TẮT'}**", ephemeral=True
    )

@tree.command(name="tw", description="Tiếng Trung phồn thể")
async def cmd_tw(interaction: discord.Interaction):
    await set_setting(interaction.user.id, "lang_mode", "zh-TW")
    await interaction.response.send_message("🇹🇼 已切換到繁體中文。", ephemeral=True)

@tree.command(name="vi", description="Tiếng Việt")
async def cmd_vi(interaction: discord.Interaction):
    await set_setting(interaction.user.id, "lang_mode", "vi")
    await interaction.response.send_message("🇻🇳 Đã chuyển về tiếng Việt.", ephemeral=True)

@tree.command(name="profile", description="Xem/đặt hồ sơ cá nhân")
@app_commands.describe(text="Nội dung hồ sơ (bỏ trống để xem)")
async def cmd_profile(interaction: discord.Interaction, text: str = ""):
    user_id = interaction.user.id
    if text:
        await set_profile(user_id, text)
        await interaction.response.send_message(f"✅ Hồ sơ đã cập nhật.", ephemeral=True)
    else:
        profile = await get_profile(user_id)
        await interaction.response.send_message(
            f"👤 _{profile}_" if profile else "👤 Chưa có hồ sơ. Dùng `/profile [mô tả]`",
            ephemeral=True,
        )

@tree.command(name="stats", description="Thống kê")
async def cmd_stats(interaction: discord.Interaction):
    user_id = interaction.user.id
    history = await get_history(user_id, limit=100)
    model_key = await get_setting(user_id, "model_key", "flash")
    auto_mode = await get_setting(user_id, "auto_mode", "1")
    user_msgs = sum(1 for m in history if m["role"] == "user")
    await interaction.response.send_message(
        f"📊 **Thống kê**\n"
        f"💬 Tin nhắn: **{user_msgs}**\n"
        f"🤖 Model: **{MODEL_REGISTRY.get(model_key,{}).get('name', model_key)}**\n"
        f"🔄 Auto: **{'BẬT' if auto_mode=='1' else 'TẮT'}**",
        ephemeral=True,
    )

@tree.command(name="tokens", description="Chi phí token")
async def cmd_tokens(interaction: discord.Interaction):
    report = await get_usage_report(interaction.user.id)
    await interaction.response.send_message(_fmt(report), ephemeral=True)

@tree.command(name="remind", description="Đặt nhắc nhở")
@app_commands.describe(text="Ví dụ: họp team 9h sáng mai")
async def cmd_remind(interaction: discord.Interaction, text: str = ""):
    if not text:
        await interaction.response.send_message(
            "Cú pháp: `/remind [nội dung] [thời gian]`", ephemeral=True); return
    await interaction.response.defer(ephemeral=True)
    result = await set_reminder_from_text(interaction.user.id, text)
    await interaction.followup.send(_fmt(result))

@tree.command(name="reminders", description="Danh sách nhắc nhở")
async def cmd_reminders(interaction: discord.Interaction):
    text = await list_reminders_text(interaction.user.id)
    await interaction.response.send_message(_fmt(text), ephemeral=True)

@tree.command(name="mn", description="Quản lý chi tiêu")
@app_commands.describe(args="+500 cà phê | -200 ăn trưa | week | all")
async def cmd_mn(interaction: discord.Interaction, args: str = ""):
    await interaction.response.defer(ephemeral=True)
    result = await handle_money_command(interaction.user.id, args)
    await interaction.followup.send(_fmt(result))

@tree.command(name="pro", description="Phân tích sâu")
@app_commands.describe(task="Câu hỏi hoặc nhiệm vụ")
async def cmd_pro(interaction: discord.Interaction, task: str = ""):
    if not task:
        await interaction.response.send_message("Dùng `/pro [câu hỏi]`", ephemeral=True); return
    await interaction.response.defer()
    result = await run_pro_workflow(interaction.user.id, task)
    parts = _split_msg(_fmt(result))
    await interaction.followup.send(parts[0])
    for p in parts[1:]: await interaction.channel.send(p)

@tree.command(name="agent", description="Agentic loop")
@app_commands.describe(task="Nhiệm vụ")
async def cmd_agent(interaction: discord.Interaction, task: str = ""):
    if not task:
        await interaction.response.send_message("Dùng `/agent [nhiệm vụ]`", ephemeral=True); return
    await interaction.response.defer()
    result = await run_agentic_loop(interaction.user.id, task)
    parts = _split_msg(_fmt(result))
    await interaction.followup.send(parts[0])
    for p in parts[1:]: await interaction.channel.send(p)

@tree.command(name="coder", description="Workflow lập trình")
@app_commands.describe(task="Yêu cầu lập trình")
async def cmd_coder(interaction: discord.Interaction, task: str = ""):
    if not task:
        await interaction.response.send_message("Dùng `/coder [yêu cầu]`", ephemeral=True); return
    await interaction.response.defer()
    result = await run_coder_workflow(interaction.user.id, task)
    parts = _split_msg(_fmt(result))
    await interaction.followup.send(parts[0])
    for p in parts[1:]: await interaction.channel.send(p)

@tree.command(name="rag", description="Quản lý tài liệu RAG")
@app_commands.describe(action="list | clear", filename="Tên file cần xóa")
async def cmd_rag(interaction: discord.Interaction, action: str = "list", filename: str = ""):
    user_id = interaction.user.id
    if action == "list":
        await interaction.response.send_message(_fmt(await list_docs(user_id)), ephemeral=True)
    elif action == "clear" and filename:
        await interaction.response.send_message(_fmt(await delete_doc(user_id, filename)), ephemeral=True)
    else:
        await interaction.response.send_message("`/rag list` | `/rag clear <file>`", ephemeral=True)

@tree.command(name="todo", description="Thêm task")
@app_commands.describe(task="Nội dung")
async def cmd_todo(interaction: discord.Interaction, task: str = ""):
    if not task:
        await interaction.response.send_message("Dùng `/todo [nội dung]`", ephemeral=True); return
    task_id = await add_task(interaction.user.id, task)
    await interaction.response.send_message(f"✅ Đã thêm task `[{task_id}]`: {task}", ephemeral=True)

@tree.command(name="tasks", description="Danh sách công việc")
async def cmd_tasks(interaction: discord.Interaction):
    await interaction.response.send_message(_fmt(await build_task_list(interaction.user.id)), ephemeral=True)

@tree.command(name="done", description="Đánh dấu hoàn thành")
@app_commands.describe(task_id="ID task")
async def cmd_done(interaction: discord.Interaction, task_id: int):
    ok = await complete_task(interaction.user.id, task_id)
    msg = f"✅ Đã hoàn thành task `[{task_id}]`." if ok else f"❌ Không tìm thấy task `[{task_id}]`."
    await interaction.response.send_message(msg, ephemeral=True)

@tree.command(name="deltask", description="Xóa task")
@app_commands.describe(task_id="ID task")
async def cmd_deltask(interaction: discord.Interaction, task_id: int):
    ok = await delete_task(interaction.user.id, task_id)
    msg = f"🗑 Đã xóa task `[{task_id}]`." if ok else f"❌ Không tìm thấy task `[{task_id}]`."
    await interaction.response.send_message(msg, ephemeral=True)

@tree.command(name="pomodoro", description="Bắt đầu Pomodoro")
async def cmd_pomodoro(interaction: discord.Interaction):
    session_id = await start_pomodoro(interaction.user.id)
    count = await get_pomodoro_count_today(interaction.user.id)
    await interaction.response.send_message(
        f"🍅 Pomodoro #{count} bắt đầu! Tập trung 25 phút nhé.", ephemeral=True)

@tree.command(name="motivation", description="Câu động lực")
async def cmd_motivation(interaction: discord.Interaction):
    await interaction.response.send_message(_fmt(get_motivation()))

@tree.command(name="checkin", description="Báo cáo ngày")
async def cmd_checkin(interaction: discord.Interaction):
    await interaction.response.send_message(_fmt(await get_daily_summary(interaction.user.id)), ephemeral=True)

@tree.command(name="user", description="Quản lý whitelist (owner only)")
@app_commands.describe(action="add | remove | list", user_id_str="Discord User ID")
async def cmd_user(interaction: discord.Interaction, action: str = "list", user_id_str: str = ""):
    if OWNER_ID != 0 and interaction.user.id != OWNER_ID:
        await interaction.response.send_message("⛔ Owner only.", ephemeral=True); return
    if action == "list":
        users = await list_allowed_users()
        await interaction.response.send_message(
            "📋 " + (", ".join(f"`{u['user_id']}`" for u in users) if users else "Chưa có user nào."),
            ephemeral=True,
        )
    elif action in ("add", "remove") and user_id_str.isdigit():
        uid = int(user_id_str)
        if action == "add":
            await add_allowed_user(uid, interaction.user.id)
            await interaction.response.send_message(f"✅ Đã thêm `{uid}`.", ephemeral=True)
        else:
            await remove_allowed_user(uid)
            await interaction.response.send_message(f"🗑 Đã xóa `{uid}`.", ephemeral=True)
    else:
        await interaction.response.send_message("`/user list|add|remove <id>`", ephemeral=True)


@tree.command(name="mapi", description="Mistral AI dashboard")
async def _mapi(interaction: discord.Interaction):
    await cmd_mapi_discord(interaction)

@tree.command(name="gapi", description="Groq Cloud dashboard")
async def _gapi(interaction: discord.Interaction):
    await cmd_gapi_discord(interaction)

@tree.command(name="gemapi", description="Google Gemini dashboard")
async def _gemapi(interaction: discord.Interaction):
    await cmd_gemapi_discord(interaction)


# ── /web /sum /quiz ──────────────────────────────────────────────────────────

@tree.command(name="web", description="Tìm kiếm web bằng Google")
@app_commands.describe(query="Câu hỏi tìm kiếm")
async def cmd_web_dc(interaction: discord.Interaction, query: str = ""):
    if not query:
        await interaction.response.send_message("Dùng `/web [câu hỏi]`", ephemeral=True); return
    await interaction.response.defer()
    from web_tools import web_search
    result = await web_search(query, interaction.user.id)
    parts = _split_msg(_fmt(result))
    await interaction.followup.send(parts[0])
    for p in parts[1:]: await interaction.channel.send(p)

@tree.command(name="sum", description="Tóm tắt nội dung URL")
@app_commands.describe(url="Link cần tóm tắt")
async def cmd_sum_dc(interaction: discord.Interaction, url: str = ""):
    if not url or not url.startswith("http"):
        await interaction.response.send_message("Dùng `/sum [URL]`", ephemeral=True); return
    await interaction.response.defer()
    from web_tools import summarize_url
    result = await summarize_url(url, interaction.user.id)
    parts = _split_msg(_fmt(result))
    await interaction.followup.send(parts[0])
    for p in parts[1:]: await interaction.channel.send(p)

@tree.command(name="quiz", description="Ôn thi iPAS AI應用規劃師")
@app_commands.describe(topic="Chủ đề (bỏ trống = random)")
async def cmd_quiz_dc(interaction: discord.Interaction, topic: str = ""):
    await interaction.response.defer()
    from web_tools import generate_quiz
    result = await generate_quiz(interaction.user.id, topic)
    await interaction.followup.send(_fmt(result))



# ── Google Services ──────────────────────────────────────────────────────────

@tree.command(name="gauth", description="Kết nối Google Account")
@app_commands.describe(code="Authorization code (bỏ trống để lấy link)")
async def cmd_gauth_dc(interaction: discord.Interaction, code: str = ""):
    from google_services import get_auth_url, exchange_code, is_connected, GOOGLE_CLIENT_ID
    if not GOOGLE_CLIENT_ID:
        await interaction.response.send_message("❌ Thiếu GOOGLE_CLIENT_ID", ephemeral=True); return
    if code:
        await interaction.response.defer(ephemeral=True)
        result = await exchange_code(interaction.user.id, code)
        await interaction.followup.send(_fmt(result))
    else:
        connected = await is_connected(interaction.user.id)
        if connected:
            await interaction.response.send_message("✅ Đã kết nối Google. Dùng /cal /gmail /gdrive", ephemeral=True)
        else:
            url = get_auth_url(state=str(interaction.user.id))
            await interaction.response.send_message(f"🔐 **Kết nối Google**\n\n1. Mở: {url}\n2. Đăng nhập & cho phép\n3. Copy code từ URL\n4. `/gauth [code]`", ephemeral=True)

@tree.command(name="cal", description="Xem/thêm lịch Google Calendar")
@app_commands.describe(args="Số ngày hoặc 'add sự kiện'")
async def cmd_cal_dc(interaction: discord.Interaction, args: str = ""):
    from google_services import list_events, add_event, is_connected
    if not await is_connected(interaction.user.id):
        await interaction.response.send_message("❌ Chưa kết nối. Dùng /gauth", ephemeral=True); return
    await interaction.response.defer()
    if args.lower().startswith("add "):
        text = args[4:].strip()
        from llm_core import _call_groq_quick
        from datetime import datetime
        import json as _json, re
        now_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M")
        prompt = (f'Parse into calendar event. Now: {now_str} UTC (Asia/Taipei UTC+8).\n'
                  f'Input: "{text}"\nReturn JSON only: {{"title":"...","start":"YYYY-MM-DDTHH:MM:SS+08:00","end":"YYYY-MM-DDTHH:MM:SS+08:00"}}')
        try:
            raw = await _call_groq_quick("Return only valid JSON.", prompt)
            m = re.search(r'\{[^{}]*\}', raw or "", re.DOTALL)
            data = _json.loads(m.group())
            result = await add_event(interaction.user.id, data["title"], data["start"], data["end"])
        except Exception as e:
            result = f"❌ Lỗi: {e}"
        await interaction.followup.send(_fmt(result))
    else:
        days = int(args) if args.isdigit() else 7
        result = await list_events(interaction.user.id, days)
        await interaction.followup.send(_fmt(result))

@tree.command(name="gmail", description="Xem email chưa đọc")
async def cmd_gmail_dc(interaction: discord.Interaction):
    from google_services import list_unread, is_connected
    if not await is_connected(interaction.user.id):
        await interaction.response.send_message("❌ Chưa kết nối. Dùng /gauth", ephemeral=True); return
    await interaction.response.defer(ephemeral=True)
    result = await list_unread(interaction.user.id)
    await interaction.followup.send(_fmt(result))

@tree.command(name="gdrive", description="Tìm file Google Drive")
@app_commands.describe(query="Từ khóa tìm kiếm")
async def cmd_gdrive_dc(interaction: discord.Interaction, query: str = ""):
    if not query:
        await interaction.response.send_message("Dùng `/gdrive [từ khóa]`", ephemeral=True); return
    from google_services import search_drive, is_connected
    if not await is_connected(interaction.user.id):
        await interaction.response.send_message("❌ Chưa kết nối. Dùng /gauth", ephemeral=True); return
    await interaction.response.defer(ephemeral=True)
    result = await search_drive(interaction.user.id, query)
    await interaction.followup.send(_fmt(result))


# ── FIX 3: Entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # bot.run() handles event loop internally — do NOT use asyncio.run()
    bot.run(DISCORD_TOKEN)
