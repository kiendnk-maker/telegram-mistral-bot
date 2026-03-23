"""
llm_core.py - Core LLM interface using Groq API and Mistral API

FIXES vs original:
  - Remove MistralClient + ChatMessage (mistralai v0.x — deleted in v1.x)
  - Remove AsyncOpenAI compat wrapper (caused Cloudflare 403 on Railway)
  - Use mistralai.Mistral SDK directly (v1.x)
  - history returns list[dict] not list[ChatMessage]
  - _call_mistral_sync uses complete_async()
"""

import os
import asyncio
import logging
from typing import Optional, AsyncGenerator

from mistralai import Mistral
from groq import AsyncGroq

from database import (
    get_history, get_summary, add_message, log_token_usage,
    get_profile, set_summary, get_setting
)
from prompts import MODEL_REGISTRY, get_system_prompt

logger = logging.getLogger(__name__)

# ── Singleton clients ─────────────────────────────────────────────────────────
_mistral_client = None  # mistralai v1.x SDK
_groq_async = None
_groq_sync = None


def _get_mistral() -> Mistral:
    global _mistral_client
    if _mistral_client is None:
        _mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    return _mistral_client


def _get_groq_async() -> AsyncGroq:
    global _groq_async
    if _groq_async is None:
        _groq_async = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_async


def _get_groq_sync():
    global _groq_sync
    if _groq_sync is None:
        from groq import Groq
        _groq_sync = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_sync


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
_SIMPLE_KW = {
    "xin chào", "hello", "hi ", "hey", "chào", "cảm ơn",
    "thanks", "thank you", "bye", "tạm biệt", "good morning", "good night",
    "oke", "ok ", "được", "vâng", "dạ", "có không", "là gì",
}


def _match(text_lower: str, keywords: set) -> bool:
    return any(k in text_lower for k in keywords)


async def resolve_model(user_id: int, text: str) -> str:
    auto_mode = await get_setting(user_id, "auto_mode", "1")
    if auto_mode != "1":
        return await get_setting(user_id, "model_key", "groq_large")
    t = text.lower()
    if _match(t, _CODE_HARD_KW): return "kimi"
    if _match(t, _MATH_KW): return "qwen3"
    if _match(t, _CODE_KW) or _match(t, _REASON_KW) or _match(t, _CREATIVE_KW): return "gpt_120b"
    if len(text) > 200: return "gpt_120b"
    return "small"


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
    full_system = system_prompt
    chat_history = []
    for m in history:
        if m["role"] == "system":
            full_system += f"\n\n{m['content']}"
        else:
            chat_history.append({"role": m["role"], "content": m["content"]})
    return [{"role": "system", "content": full_system}] + chat_history


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

    model_id = MODEL_REGISTRY[model_key]["model_id"]
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
    messages = _build_messages(system_prompt, history)

    if not save_history:
        while messages and messages[-1]["role"] == "assistant":
            messages.pop()
        messages.append({"role": "user", "content": user_message})

    provider = MODEL_REGISTRY[model_key].get("provider", "mistral")
    full_reply = ""
    thinking_enabled = MODEL_REGISTRY[model_key].get("thinking", False)

    if provider == "groq":
        stream = await _get_groq_async().chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=8000 if thinking_enabled else 1024,
            temperature=0.6 if thinking_enabled else 0.7,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_reply += delta
                yield delta, model_key
    else:
        # mistralai v1.x — no Cloudflare block on Railway
        stream = await _get_mistral().chat.stream_async(
            model=model_id,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
        async for event in stream:
            delta = event.data.choices[0].delta.content or ""
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


# ── Vision ────────────────────────────────────────────────────────────────────

_VISION_MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"
_VISION_MODEL_KEY = "llama4"


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
    messages = [{"role": "system", "content": lang_instr}] + vision_messages

    full_reply = ""
    stream = await _get_groq_async().chat.completions.create(
        model=_VISION_MODEL_ID,
        messages=messages,
        max_tokens=800,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            full_reply += delta
            yield delta, _VISION_MODEL_KEY

    if full_reply:
        await log_token_usage(user_id, _VISION_MODEL_KEY, 0, len(full_reply.split()))


# ── Mistral OCR ───────────────────────────────────────────────────────────────

async def call_ocr_mistral(user_id: int, image_base64: str) -> str:
    import httpx
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.mistral.ai/v1/ocr",
            headers={
                "Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": "mistral-ocr-latest",
                "document": {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{image_base64}",
                },
            },
        )
        response.raise_for_status()
        data = response.json()
    pages = data.get("pages", [])
    if not pages:
        return "Không tìm thấy văn bản trong ảnh."
    text = "\n\n".join(p.get("markdown", "") for p in pages).strip()
    await log_token_usage(user_id, "vision", 0, len(text.split()))
    return text or "Không tìm thấy văn bản trong ảnh."


# ── Audio transcription ───────────────────────────────────────────────────────

async def transcribe_audio(audio_path: str, language: str = "vi") -> str:
    def _run():
        with open(audio_path, "rb") as f:
            transcription = _get_groq_sync().audio.transcriptions.create(
                file=f,
                model="whisper-large-v3-turbo",
                language=language,
            )
        return transcription.text
    return await asyncio.get_event_loop().run_in_executor(None, _run)


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _summarize_messages(messages: list[dict]) -> str:
    text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    response = await _get_groq_async().chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": f"Tóm tắt ngắn gọn cuộc hội thoại này (tối đa 100 từ):\n{text}"}],
        max_tokens=200,
    )
    return response.choices[0].message.content


async def _call_mistral_sync(system: str, user: str) -> str:
    """One-shot Mistral call (used by reminder_system)."""
    response = await _get_mistral().chat.complete_async(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=256,
    )
    return response.choices[0].message.content
