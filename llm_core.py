"""
llm_core.py - Core LLM interface using Groq + Mistral APIs
"""

import os
import base64
import logging
import asyncio
from typing import Optional, AsyncGenerator

from groq import AsyncGroq
from mistralai import Mistral

from database import (
    get_history, get_summary, add_message, log_token_usage,
    get_profile, set_summary, get_setting
)
from prompts import MODEL_REGISTRY, get_system_prompt

logger = logging.getLogger(__name__)

# ── Singleton clients ─────────────────────────────────────────────────────────
_groq_client: Optional[AsyncGroq] = None
_mistral_client: Optional[Mistral] = None


def _get_groq() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


def _get_mistral() -> Mistral:
    global _mistral_client
    if _mistral_client is None:
        _mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    return _mistral_client


# ── Model routing ─────────────────────────────────────────────────────────────

_CODE_HARD_KW = {
    "dynamic programming", "quy hoạch động", "competitive programming",
    "system design", "thiết kế hệ thống", "design pattern",
    "tối ưu hóa thuật toán", "optimize algorithm", "complexity analysis",
    "big o", "data structure", "cấu trúc dữ liệu phức tạp",
    "graph algorithm", "thuật toán đồ thị", "backtracking",
    "viết compiler", "build framework", "kiến trúc hệ thống",
}
_MATH_KW = {
    "tính ", "toán ", "math", "calculate", "equation", "formula",
    "xác suất", "thống kê", "probability", "statistics", "calculus",
    "đạo hàm", "tích phân", "ma trận", "vector", "lượng giác",
    "bài toán", "giải phương trình", "chứng minh", "định lý",
    "physics", "vật lý", "chemistry", "hóa học", "số học",
}
_CODE_KW = {
    "code", "viết hàm", "debug", "lỗi code", "python", "javascript",
    "typescript", "golang", "rust", "java", "c++", "c#", "kotlin", "swift",
    "function", "class", "import", "def ", "bug", "syntax", "script",
    "lập trình", "compile", "runtime error", "api", "database", "sql",
    "query", "regex", "git ", "docker", "bash", "shell", "json",
    "react", "vue", "django", "flask", "fastapi", "async",
}
_REASON_KW = {
    "tại sao", "phân tích", "so sánh", "giải thích chi tiết", "hãy giải thích",
    "why", "analyze", "compare", "explain", "evaluate", "đánh giá",
    "ưu nhược", "pros and cons", "chiến lược", "strategy", "nhận xét",
    "luận điểm", "nguyên nhân", "hậu quả", "ảnh hưởng", "tác động",
    "quan điểm", "nghiên cứu", "review", "assessment",
}
_CREATIVE_KW = {
    "viết bài", "viết đoạn", "viết thư", "viết email", "sáng tác",
    "write a", "write an", "essay", "paragraph", "story", "poem",
    "thơ", "truyện", "kịch bản", "báo cáo", "proposal", "cover letter",
}


def _match(text_lower: str, keywords: set) -> bool:
    return any(k in text_lower for k in keywords)


async def resolve_model(user_id: int, text: str) -> str:
    auto_mode = await get_setting(user_id, "auto_mode", "1")
    if auto_mode != "1":
        return await get_setting(user_id, "model_key", "flash")
    t = text.lower()
    if _match(t, _CODE_HARD_KW) or _match(t, _MATH_KW):
        return "pro"
    if _match(t, _CODE_KW) or _match(t, _REASON_KW) or _match(t, _CREATIVE_KW):
        return "flash_think"
    if len(text) > 200:
        return "flash_think"
    return "flash"


# ── History management ────────────────────────────────────────────────────────

async def get_history_with_summary(user_id: int) -> list[dict]:
    history = await get_history(user_id, limit=30)
    if len(history) > 20:
        old_msgs = history[:-10]
        try:
            summary_text = await _summarize_messages(old_msgs)
            await set_summary(user_id, summary_text)
        except Exception as e:
            logger.warning(f"Auto-summarize failed for user {user_id}: {e}")
        history = history[-10:]
    summary = await get_summary(user_id)
    messages: list[dict] = []
    if summary:
        messages.append({"role": "system", "content": f"Tóm tắt hội thoại trước: {summary}"})
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    return messages


def _build_messages(system_prompt: str, history: list[dict]) -> list[dict]:
    """Convert history list[dict] to OpenAI-format messages with system prompt."""
    messages = [{"role": "system", "content": system_prompt}]
    for m in history:
        if m["role"] == "system":
            messages[0]["content"] += f"\n\n{m['content']}"
        else:
            messages.append({"role": m["role"], "content": m["content"]})
    return messages


# ── Streaming chat ────────────────────────────────────────────────────────────

async def call_llm_stream(
    user_id: int,
    user_message: str,
    model_key: Optional[str] = None,
    extra_context: Optional[str] = None,
    save_history: bool = True,
) -> AsyncGenerator[tuple[str, str], None]:
    if model_key is None:
        model_key = await resolve_model(user_id, user_message)

    reg = MODEL_REGISTRY[model_key]
    model_id = reg["model_id"]
    provider = reg["provider"]
    profile = await get_profile(user_id)
    system_prompt = get_system_prompt(model_key, profile)

    lang_mode = await get_setting(user_id, "lang_mode", "vi")
    if lang_mode == "zh-TW":
        system_prompt += "\n\n【強制規則】無論使用者用什麼語言，你必須用繁體中文回覆所有訊息。"
    else:
        system_prompt += "\n\nQuy tắc bắt buộc: Luôn trả lời bằng tiếng Việt, bất kể người dùng viết bằng ngôn ngữ gì."

    if extra_context:
        system_prompt += f"\n\nContext từ tài liệu:\n{extra_context}"

    if save_history:
        await add_message(user_id, "user", user_message)

    history = await get_history_with_summary(user_id)

    if not save_history:
        # Remove trailing assistant messages and append user message manually
        while history and history[-1]["role"] == "assistant":
            history.pop()
        history.append({"role": "user", "content": user_message})

    messages = _build_messages(system_prompt, history)

    full_reply = ""

    if provider == "groq":
        stream = await _get_groq().chat.completions.create(
            model=model_id,
            messages=messages,
            stream=True,
            max_tokens=2048,
            temperature=0.7,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_reply += delta
                yield delta, model_key

    elif provider == "mistral":
        async for chunk in await _get_mistral().chat.stream_async(
            model=model_id,
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
        ):
            delta = chunk.data.choices[0].delta.content or ""
            if delta:
                full_reply += delta
                yield delta, model_key

    if save_history:
        await add_message(user_id, "assistant", full_reply)


# ── Non-streaming chat ────────────────────────────────────────────────────────

async def call_llm(
    user_id: int,
    user_message: str,
    model_key: Optional[str] = None,
    extra_context: Optional[str] = None,
) -> tuple[str, str]:
    full_reply = ""
    async for chunk, mk in call_llm_stream(user_id, user_message, model_key, extra_context):
        full_reply += chunk
        model_key = mk
    return full_reply, model_key


# ── Vision (image understanding) ─────────────────────────────────────────────

_VISION_MODEL_ID = "pixtral-12b-2409"
_VISION_MODEL_KEY = "flash"


async def call_vision_stream(
    user_id: int,
    vision_messages: list[dict],
) -> AsyncGenerator[tuple[str, str], None]:
    lang_mode = await get_setting(user_id, "lang_mode", "vi")
    lang_instr = (
        "無論使用者用什麼語言，必須用繁體中文回覆所有訊息。"
        if lang_mode == "zh-TW"
        else "Luôn trả lời bằng tiếng Việt, bất kể người dùng viết bằng ngôn ngữ gì."
    )

    # Build Mistral-compatible messages
    mistral_messages = []
    system_parts = [lang_instr]

    for msg in vision_messages:
        if msg["role"] == "system":
            system_parts.append(msg["content"] if isinstance(msg["content"], str) else "")
            continue
        role = "assistant" if msg["role"] == "assistant" else "user"
        content = msg["content"]
        if isinstance(content, str):
            mistral_messages.append({"role": role, "content": content})
        else:
            parts = []
            for part in content:
                if part["type"] == "text":
                    parts.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image_url":
                    url = part["image_url"]
                    if isinstance(url, dict):
                        url = url.get("url", "")
                    parts.append({"type": "image_url", "image_url": url})
            mistral_messages.append({"role": role, "content": parts})

    # Prepend system as first user message if no messages yet, or inject into system
    if system_parts:
        sys_text = "\n".join(system_parts)
        if mistral_messages and mistral_messages[0]["role"] == "user":
            first = mistral_messages[0]
            if isinstance(first["content"], str):
                first["content"] = sys_text + "\n\n" + first["content"]
            elif isinstance(first["content"], list):
                first["content"].insert(0, {"type": "text", "text": sys_text})

    full_reply = ""
    async for chunk in await _get_mistral().chat.stream_async(
        model=_VISION_MODEL_ID,
        messages=mistral_messages,
        max_tokens=1024,
    ):
        delta = chunk.data.choices[0].delta.content or ""
        if delta:
            full_reply += delta
            yield delta, _VISION_MODEL_KEY

    if full_reply:
        await log_token_usage(user_id, _VISION_MODEL_KEY, 0, len(full_reply.split()))


# ── OCR using Mistral pixtral ─────────────────────────────────────────────────

async def call_ocr_mistral(user_id: int, image_base64: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract all text from this image. Return only the extracted text in its original language and formatting, nothing else."},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"},
        ],
    }]
    response = await _get_mistral().chat.complete_async(
        model=_VISION_MODEL_ID,
        messages=messages,
        max_tokens=2048,
    )
    text = (response.choices[0].message.content or "").strip()
    await log_token_usage(user_id, "vision", 0, len(text.split()))
    return text or "Không tìm thấy văn bản trong ảnh."


# ── Audio transcription ───────────────────────────────────────────────────────

async def transcribe_audio(audio_path: str, language: str = "vi") -> str:
    ext = audio_path.rsplit(".", 1)[-1].lower()
    mime_map = {
        "ogg": "audio/ogg",
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "m4a": "audio/mp4",
    }
    mime_type = mime_map.get(ext, "audio/ogg")

    with open(audio_path, "rb") as f:
        transcription = await _get_groq().audio.transcriptions.create(
            file=(os.path.basename(audio_path), f, mime_type),
            model="whisper-large-v3",
            language=language,
        )
    return transcription.text


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _summarize_messages(messages: list[dict]) -> str:
    text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    resp = await _get_groq().chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": f"Tóm tắt ngắn gọn cuộc hội thoại này (tối đa 100 từ):\n{text}"},
        ],
        max_tokens=200,
        temperature=0.3,
    )
    return resp.choices[0].message.content or ""


async def _call_groq_quick(system: str, user: str) -> str:
    """One-shot Groq call (used by reminder_system)."""
    resp = await _get_groq().chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=256,
        temperature=0.3,
    )
    return resp.choices[0].message.content or ""


# Backward-compat alias used by reminder_system
_call_mistral_sync = _call_groq_quick
