"""
llm_core.py - Core LLM interface using Mistral API and Groq API
"""
import os
import asyncio
import logging
from typing import Optional, AsyncGenerator

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from openai import AsyncOpenAI
from groq import AsyncGroq

from database import (
    get_history, get_summary, add_message, log_token_usage,
    get_profile, set_summary, get_setting
)
from prompts import MODEL_REGISTRY, get_system_prompt

logger = logging.getLogger(__name__)

# ── Singleton clients ─────────────────────────────────────────────────────────

_mistral_client = None       # sync, for summarize/embed helpers
_mistral_async  = None       # async OpenAI-compat, for streaming chat
_groq_async     = None       # async Groq, for streaming chat
_groq_sync      = None       # sync Groq, for audio transcription


def _get_mistral():
    global _mistral_client
    if _mistral_client is None:
        _mistral_client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
    return _mistral_client


def _get_mistral_async() -> AsyncOpenAI:
    global _mistral_async
    if _mistral_async is None:
        _mistral_async = AsyncOpenAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            base_url="https://api.mistral.ai/v1",
        )
    return _mistral_async


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

_CODE_KW = {
    "code", "viết hàm", "debug", "lỗi code", "python", "javascript",
    "typescript", "golang", "rust", "java", "c++", "c#", "kotlin", "swift",
    "function", "class", "import", "def ", "bug", "syntax", "algorithm",
    "thuật toán", "script", "lập trình", "compile", "runtime error",
    "stack overflow", "api", "database", "sql", "query", "regex",
    "git ", "docker", "terminal", "bash", "shell", "json", "html", "css",
    "react", "vue", "django", "flask", "fastapi", "async", "coroutine",
}

_REASON_KW = {
    "tại sao", "phân tích", "so sánh", "giải thích chi tiết", "hãy giải thích",
    "why", "analyze", "compare", "explain", "evaluate", "đánh giá",
    "ưu nhược", "pros and cons", "chiến lược", "strategy", "nhận xét",
    "bình luận", "luận điểm", "lý do", "nguyên nhân", "hậu quả",
    "ảnh hưởng", "tác động", "quan điểm", "triết học", "nghiên cứu",
    "review", "critique", "assessment", "implication",
}

_MATH_KW = {
    "tính ", "toán ", "math", "calculate", "equation", "formula",
    "xác suất", "thống kê", "probability", "statistics", "calculus",
    "đạo hàm", "tích phân", "ma trận", "vector", "lượng giác",
    "bài toán", "giải phương trình", "chứng minh", "định lý",
}

_CREATIVE_KW = {
    "viết bài", "viết đoạn", "viết thư", "viết email", "sáng tác",
    "write a", "write an", "essay", "paragraph", "story", "poem",
    "thơ", "truyện", "kịch bản", "script", "tóm tắt dài", "báo cáo",
    "proposal", "cover letter", "presentation",
}

# Short greetings / simple queries — fast model is fine
_SIMPLE_KW = {
    "xin chào", "hello", "hi", "hey", "chào", "oke", "ok", "cảm ơn",
    "thanks", "thank you", "bye", "tạm biệt", "good morning", "good night",
}


def _match(text_lower: str, keywords: set) -> bool:
    return any(k in text_lower for k in keywords)


async def resolve_model(user_id: int, text: str) -> str:
    """
    Smart provider-agnostic auto-router.
    Priority: code → kimi | math → qwen3 | reason/long → groq_large | else → groq_fast
    Respects user's manual model choice when auto_mode is off.
    """
    auto_mode = await get_setting(user_id, "auto_mode", "1")
    if auto_mode != "1":
        return await get_setting(user_id, "model_key", "groq_fast")

    t = text.lower()
    length = len(text)

    # 1. Code → best code model
    if _match(t, _CODE_KW):
        return "kimi"

    # 2. Math → qwen3 (strong at math/science)
    if _match(t, _MATH_KW):
        return "qwen3"

    # 3. Deep reasoning / analysis / creative writing → GPT OSS 120B
    #    ($0.15/$0.60 vs groq_large $0.59/$0.79 — 4x cheaper, comparable quality)
    if _match(t, _REASON_KW) or _match(t, _CREATIVE_KW):
        return "gpt_120b"

    # 4. Long message (>250 chars) → likely complex, use gpt_120b
    if length > 250:
        return "gpt_120b"

    # 5. Simple greeting / very short → fast model
    if _match(t, _SIMPLE_KW) or length < 30:
        return "groq_fast"

    # 6. Default: fast model (good enough for most chat)
    return "groq_fast"


# ── History management ────────────────────────────────────────────────────────

async def get_history_with_summary(user_id: int) -> list[ChatMessage]:
    """Get chat history, auto-summarize if too long."""
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
    messages: list[ChatMessage] = []

    if summary:
        messages.append(ChatMessage(role="system", content=f"Tóm tắt hội thoại trước: {summary}"))

    for h in history:
        messages.append(ChatMessage(role=h["role"], content=h["content"]))

    return messages


def _build_messages(system_prompt: str, history: list[ChatMessage]) -> list[dict]:
    """Convert system_prompt + ChatMessage history to plain dict list.
    Merges any summary system message into the system_prompt."""
    full_system = system_prompt
    chat_history = []
    for m in history:
        if m.role == "system":
            full_system += f"\n\n{m.content}"
        else:
            chat_history.append({"role": m.role, "content": m.content})
    return [{"role": "system", "content": full_system}] + chat_history


# ── Streaming chat ────────────────────────────────────────────────────────────

async def call_llm_stream(
    user_id: int,
    user_message: str,
    model_key: Optional[str] = None,
    extra_context: Optional[str] = None,
) -> AsyncGenerator[tuple[str, str], None]:
    """
    Async generator: yields (text_chunk, model_key).
    Saves assistant reply to DB after all chunks are yielded.
    """
    if model_key is None:
        model_key = await resolve_model(user_id, user_message)

    model_id = MODEL_REGISTRY[model_key]["model_id"]
    profile = await get_profile(user_id)
    system_prompt = get_system_prompt(model_key, profile)

    if extra_context:
        system_prompt += f"\n\nContext từ tài liệu:\n{extra_context}"

    await add_message(user_id, "user", user_message)
    history = await get_history_with_summary(user_id)
    messages = _build_messages(system_prompt, history)
    provider = MODEL_REGISTRY[model_key].get("provider", "mistral")
    full_reply = ""

    if provider == "groq":
        stream = await _get_groq_async().chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_reply += delta
                yield delta, model_key
    else:
        stream = await _get_mistral_async().chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_reply += delta
                yield delta, model_key

    await add_message(user_id, "assistant", full_reply)


# ── Non-streaming chat (used by agents_workflow internally) ───────────────────

async def call_llm(
    user_id: int,
    user_message: str,
    model_key: Optional[str] = None,
    extra_context: Optional[str] = None,
) -> tuple[str, str]:
    """Call LLM, return (reply_text, model_key_used). Non-streaming."""
    full_reply = ""
    async for chunk, mk in call_llm_stream(user_id, user_message, model_key, extra_context):
        full_reply += chunk
        model_key = mk
    return full_reply, model_key


# ── Vision ────────────────────────────────────────────────────────────────────

async def call_vision(user_id: int, image_base64: str, prompt: str) -> str:
    """Process image with Pixtral vision model via OpenAI-compatible endpoint."""
    response = await _get_mistral_async().chat.completions.create(
        model="pixtral-large-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=1024,
    )

    reply = response.choices[0].message.content

    usage = response.usage
    if usage:
        await log_token_usage(
            user_id, "vision",
            usage.prompt_tokens,
            usage.completion_tokens,
        )

    return reply


# ── Audio transcription ───────────────────────────────────────────────────────

async def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file using Groq Whisper."""
    def _run():
        with open(audio_path, "rb") as f:
            transcription = _get_groq_sync().audio.transcriptions.create(
                file=f,
                model="whisper-large-v3-turbo",
                language="vi"
            )
        return transcription.text

    return await asyncio.get_event_loop().run_in_executor(None, _run)


# ── Internal helpers ───────────────────────────────────────────────────────────

async def _summarize_messages(messages: list[dict]) -> str:
    """Summarize a list of messages using Groq (fast + cheap)."""
    text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    response = await _get_groq_async().chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": f"Tóm tắt ngắn gọn cuộc hội thoại này (tối đa 100 từ):\n{text}"}
        ],
        max_tokens=200,
    )
    return response.choices[0].message.content


async def _call_mistral_sync(system: str, user: str) -> str:
    """Simple one-shot Mistral call (used by reminder_system for NLP parsing)."""
    def _run():
        return _get_mistral().chat(
            model="mistral-small-latest",
            messages=[
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=user),
            ],
            max_tokens=256,
        )

    response = await asyncio.get_event_loop().run_in_executor(None, _run)
    return response.choices[0].message.content
