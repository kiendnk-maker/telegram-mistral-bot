"""
llm_core.py - Core LLM interface using Mistral API
"""
import os
import asyncio
import logging
from typing import Optional

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from database import (
    get_history, get_summary, add_message, log_token_usage,
    get_profile, set_summary, get_setting
)
from prompts import MODEL_REGISTRY, get_system_prompt

logger = logging.getLogger(__name__)

mistral = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))


# ── Model routing ─────────────────────────────────────────────────────────────

CODE_KEYWORDS = [
    "code", "viết hàm", "debug", "lỗi code", "python", "javascript",
    "typescript", "function", "class", "import", "def ", "bug",
    "syntax", "algorithm", "thuật toán", "script", "lập trình",
    "compile", "runtime error", "stack overflow",
]
REASON_KEYWORDS = [
    "tại sao", "phân tích", "so sánh", "giải thích chi tiết",
    "why", "analyze", "compare", "explain", "evaluate", "đánh giá",
    "ưu nhược", "pros and cons", "chiến lược", "strategy",
]
VISION_KEYWORDS = [
    "ảnh này", "hình này", "what is in", "describe the image",
    "trong ảnh", "nhìn vào",
]


async def resolve_model(user_id: int, text: str) -> str:
    """Auto-route to best model based on content and user settings."""
    setting = await get_setting(user_id, "model_key", "small")
    auto_mode = await get_setting(user_id, "auto_mode", "1")

    if auto_mode != "1":
        return setting

    text_lower = text.lower()

    if any(k in text_lower for k in CODE_KEYWORDS):
        return "codestral"
    if any(k in text_lower for k in REASON_KEYWORDS):
        return "large"
    return "small"


# ── History management ────────────────────────────────────────────────────────

async def get_history_with_summary(user_id: int) -> list[ChatMessage]:
    """Get chat history, auto-summarize if too long."""
    history = await get_history(user_id, limit=30)

    if len(history) > 20:
        # Summarize old messages, keep last 10
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


# ── Main chat function ────────────────────────────────────────────────────────

async def call_llm(
    user_id: int,
    user_message: str,
    model_key: Optional[str] = None,
    extra_context: Optional[str] = None,
) -> tuple[str, str]:
    """
    Call LLM for a user message.
    Returns (reply_text, model_key_used).
    """
    if model_key is None:
        model_key = await resolve_model(user_id, user_message)

    model_id = MODEL_REGISTRY[model_key]["model_id"]
    profile = await get_profile(user_id)
    system_prompt = get_system_prompt(model_key, profile)

    # Inject RAG context if available
    if extra_context:
        system_prompt += f"\n\nContext từ tài liệu:\n{extra_context}"

    await add_message(user_id, "user", user_message)
    history = await get_history_with_summary(user_id)

    messages = [ChatMessage(role="system", content=system_prompt)] + history

    def _run():
        return mistral.chat(
            model=model_id,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )

    response = await asyncio.get_event_loop().run_in_executor(None, _run)

    reply = response.choices[0].message.content
    await add_message(user_id, "assistant", reply)

    # Log token usage
    usage = response.usage
    if usage:
        await log_token_usage(
            user_id, model_key,
            usage.prompt_tokens,
            usage.completion_tokens
        )

    return reply, model_key


# ── Vision ───────────────────────────────────────────────────────────────────

async def call_vision(user_id: int, image_base64: str, prompt: str) -> str:
    """Process image with Pixtral vision model."""
    messages = [
        ChatMessage(role="user", content=[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            {"type": "text", "text": prompt},
        ])
    ]

    def _run():
        return mistral.chat(
            model="pixtral-large-latest",
            messages=messages,
            max_tokens=1024,
        )

    response = await asyncio.get_event_loop().run_in_executor(None, _run)
    reply = response.choices[0].message.content

    # Log usage
    usage = response.usage
    if usage:
        await log_token_usage(user_id, "vision", usage.prompt_tokens, usage.completion_tokens)

    return reply


# ── Audio transcription ───────────────────────────────────────────────────────

async def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file using Groq Whisper."""
    from groq import Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def _run():
        with open(audio_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=f,
                model="whisper-large-v3",
                language="vi"
            )
        return transcription.text

    return await asyncio.get_event_loop().run_in_executor(None, _run)


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _summarize_messages(messages: list[dict]) -> str:
    """Summarize a list of messages into a short text."""
    text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    def _run():
        return mistral.chat(
            model="mistral-small-latest",
            messages=[
                ChatMessage(
                    role="user",
                    content=f"Tóm tắt ngắn gọn cuộc hội thoại này (tối đa 100 từ):\n{text}"
                )
            ],
            max_tokens=200,
        )

    response = await asyncio.get_event_loop().run_in_executor(None, _run)
    return response.choices[0].message.content


async def _call_mistral_sync(system: str, user: str) -> str:
    """Simple one-shot Mistral call (used by reminder_system for NLP parsing)."""
    def _run():
        return mistral.chat(
            model="mistral-small-latest",
            messages=[
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=user),
            ],
            max_tokens=256,
        )

    response = await asyncio.get_event_loop().run_in_executor(None, _run)
    return response.choices[0].message.content
