"""
llm_core.py - Core LLM interface using Google Gemini API
"""

import os
import base64
import logging
from typing import Optional, AsyncGenerator

from google import genai
from google.genai import types

from database import (
    get_history, get_summary, add_message, log_token_usage,
    get_profile, set_summary, get_setting
)
from prompts import MODEL_REGISTRY, get_system_prompt

logger = logging.getLogger(__name__)

# ── Singleton client ──────────────────────────────────────────────────────────
_gemini_client: Optional[genai.Client] = None


def _get_gemini() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _gemini_client


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


def _build_gemini_contents(system_prompt: str, history: list[dict]) -> tuple[str, list]:
    """Convert OpenAI-style history to Gemini contents + merged system instruction."""
    full_system = system_prompt
    contents = []
    for m in history:
        if m["role"] == "system":
            full_system += f"\n\n{m['content']}"
        elif m["role"] == "assistant":
            contents.append(types.Content(
                role="model",
                parts=[types.Part.from_text(text=m["content"])],
            ))
        else:
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=m["content"])],
            ))
    return full_system, contents


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
    full_system, contents = _build_gemini_contents(system_prompt, history)

    if not save_history:
        while contents and contents[-1].role == "model":
            contents.pop()
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        ))

    thinking_enabled = MODEL_REGISTRY[model_key].get("thinking", False)
    config = types.GenerateContentConfig(
        system_instruction=full_system,
        max_output_tokens=8192 if thinking_enabled else 2048,
        temperature=0.7,
        thinking_config=types.ThinkingConfig(thinking_budget=8000) if thinking_enabled else None,
    )

    full_reply = ""
    stream = await _get_gemini().aio.models.generate_content_stream(
        model=model_id,
        contents=contents,
        config=config,
    )
    async for chunk in stream:
        delta = chunk.text or ""
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

_VISION_MODEL_ID = "gemini-2.0-flash"
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

    contents = []
    system_parts = [lang_instr]
    for msg in vision_messages:
        if msg["role"] == "system":
            system_parts.append(msg["content"] if isinstance(msg["content"], str) else "")
            continue
        role = "model" if msg["role"] == "assistant" else "user"
        content = msg["content"]
        if isinstance(content, str):
            parts = [types.Part.from_text(text=content)]
        else:
            parts = []
            for part in content:
                if part["type"] == "text":
                    parts.append(types.Part.from_text(text=part["text"]))
                elif part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:"):
                        header, data = url.split(",", 1)
                        mime_type = header.split(":")[1].split(";")[0]
                        parts.append(types.Part.from_bytes(
                            data=base64.b64decode(data),
                            mime_type=mime_type,
                        ))
        contents.append(types.Content(role=role, parts=parts))

    full_reply = ""
    stream = await _get_gemini().aio.models.generate_content_stream(
        model=_VISION_MODEL_ID,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction="\n".join(system_parts),
            max_output_tokens=1024,
        ),
    )
    async for chunk in stream:
        delta = chunk.text or ""
        if delta:
            full_reply += delta
            yield delta, _VISION_MODEL_KEY

    if full_reply:
        await log_token_usage(user_id, _VISION_MODEL_KEY, 0, len(full_reply.split()))


# ── OCR (replaces Mistral OCR) ────────────────────────────────────────────────

async def call_ocr_mistral(user_id: int, image_base64: str) -> str:
    image_part = types.Part.from_bytes(
        data=base64.b64decode(image_base64),
        mime_type="image/jpeg",
    )
    response = await _get_gemini().aio.models.generate_content(
        model="gemini-2.0-flash",
        contents=[types.Content(role="user", parts=[
            image_part,
            types.Part.from_text(text=
                "Extract all text from this image. Return only the extracted text "
                "in its original language and formatting, nothing else."
            ),
        ])],
        config=types.GenerateContentConfig(max_output_tokens=2048),
    )
    text = (response.text or "").strip()
    await log_token_usage(user_id, "vision", 0, len(text.split()))
    return text or "Không tìm thấy văn bản trong ảnh."


# ── Audio transcription ───────────────────────────────────────────────────────

async def transcribe_audio(audio_path: str, language: str = "vi") -> str:
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    ext = audio_path.rsplit(".", 1)[-1].lower()
    mime_map = {
        "ogg": "audio/ogg",
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "m4a": "audio/mp4",
    }
    mime_type = mime_map.get(ext, "audio/ogg")
    lang_str = "Vietnamese" if language == "vi" else ("Chinese" if language == "zh" else language)

    response = await _get_gemini().aio.models.generate_content(
        model="gemini-2.0-flash",
        contents=[types.Content(role="user", parts=[
            types.Part.from_bytes(data=audio_data, mime_type=mime_type),
            types.Part.from_text(text=
                f"Transcribe this audio in {lang_str}. Return only the transcribed text, nothing else."
            ),
        ])],
        config=types.GenerateContentConfig(max_output_tokens=1024),
    )
    return (response.text or "").strip()


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _summarize_messages(messages: list[dict]) -> str:
    text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    response = await _get_gemini().aio.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=
            f"Tóm tắt ngắn gọn cuộc hội thoại này (tối đa 100 từ):\n{text}"
        )])],
        config=types.GenerateContentConfig(max_output_tokens=200),
    )
    return response.text or ""


async def _call_gemini_quick(system: str, user: str) -> str:
    """One-shot Gemini call (used by reminder_system)."""
    response = await _get_gemini().aio.models.generate_content(
        model="gemini-2.0-flash",
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=user)])],
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=256,
        ),
    )
    return response.text or ""


# Backward-compat alias used by reminder_system
_call_mistral_sync = _call_gemini_quick
