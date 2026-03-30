"""
web_tools.py — /web (DuckDuckGo Search), /sum (URL summary), /quiz (iPAS quiz engine)
"""

import os
import re
import json
import random
import logging
from typing import Optional

import httpx
from groq import AsyncGroq

from database import get_setting

logger = logging.getLogger(__name__)

_groq_client: Optional[AsyncGroq] = None


def _get_groq() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


async def _groq_chat(system: str, user: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
    resp = await _get_groq().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


# ── /web — DuckDuckGo Search + Groq synthesis ─────────────────────────────────

async def web_search(query: str, user_id: int = 0) -> str:
    lang_mode = await get_setting(user_id, "lang_mode", "vi") if user_id else "vi"
    lang_instr = "用繁體中文回答。" if lang_mode == "zh-TW" else "Trả lời bằng tiếng Việt."
    try:
        # Fetch DuckDuckGo instant answer API
        ddg_url = f"https://api.duckduckgo.com/?q={httpx.URL(query)}&format=json&no_html=1&skip_disambig=1"
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
                headers={"User-Agent": "Mozilla/5.0 (compatible; UltraBolt/1.0)"},
            )
            resp.raise_for_status()
            data = resp.json()

        # Extract useful content from DuckDuckGo response
        snippets = []
        if data.get("AbstractText"):
            snippets.append(data["AbstractText"])
        if data.get("Answer"):
            snippets.append(f"Answer: {data['Answer']}")
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                snippets.append(topic["Text"])

        if snippets:
            context = "\n\n".join(snippets[:6])
            answer = await _groq_chat(
                f"Tìm kiếm web và trả lời chính xác, có nguồn. Ngắn gọn. {lang_instr}",
                f"Câu hỏi: {query}\n\nThông tin tìm được:\n{context}\n\nTrả lời:",
                max_tokens=2048,
                temperature=0.3,
            )
        else:
            # Fallback: ask Groq directly with knowledge
            answer = await _groq_chat(
                f"Trả lời câu hỏi dựa trên kiến thức của bạn. Ngắn gọn, chính xác. {lang_instr}",
                f"Câu hỏi: {query}",
                max_tokens=2048,
                temperature=0.5,
            )

        return answer or "Không tìm thấy kết quả."
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"❌ Lỗi tìm kiếm: {e}"


# ── /sum — Summarize URL ─────────────────────────────────────────────────────

async def summarize_url(url: str, user_id: int = 0) -> str:
    lang_mode = await get_setting(user_id, "lang_mode", "vi") if user_id else "vi"
    lang_instr = "用繁體中文摘要。" if lang_mode == "zh-TW" else "Tóm tắt bằng tiếng Việt."
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; UltraBolt/1.0)"})
            resp.raise_for_status()
            content = resp.text
    except Exception as e:
        return f"❌ Không thể tải URL: {e}"
    clean = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', content, flags=re.IGNORECASE)
    clean = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'<[^>]+>', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()[:12000]
    if len(clean) < 50:
        return "❌ Trang web không có đủ nội dung text."
    try:
        answer = await _groq_chat(
            f"Tóm tắt ngắn gọn, rõ ràng. Bỏ quảng cáo/menu/footer. {lang_instr}",
            f"Tóm tắt nội dung này:\n\n{clean}",
            max_tokens=1024,
            temperature=0.3,
        )
        return answer or "Không thể tóm tắt."
    except Exception as e:
        logger.error(f"Summarize error: {e}")
        return f"❌ Lỗi tóm tắt: {e}"


# ── /quiz — iPAS Quiz Engine ─────────────────────────────────────────────────

_quiz_state: dict[int, dict] = {}

QUIZ_TOPICS = [
    "Human-in-the-loop, Human-over-the-loop, Human-out-of-the-loop",
    "Feature Engineering, ETL pipeline",
    "Regularization (L1/L2), Overfitting, Underfitting",
    "Bias-Variance Tradeoff",
    "Supervised vs Unsupervised vs Reinforcement Learning",
    "Neural Network, CNN, RNN, Transformer",
    "NLP, Tokenization, Word Embedding",
    "Computer Vision, Object Detection",
    "Model Evaluation: Accuracy, Precision, Recall, F1",
    "AI Ethics, Fairness, Transparency, Accountability",
    "Data Preprocessing, Missing Values, Normalization",
    "Decision Tree, Random Forest, XGBoost",
    "K-Means, DBSCAN, Clustering",
    "Generative AI, LLM, Prompt Engineering",
    "AI Project Lifecycle, CRISP-DM",
]


async def generate_quiz(user_id: int, topic: str = "") -> str:
    lang_mode = await get_setting(user_id, "lang_mode", "vi")
    lang_instr = "用繁體中文出題。" if lang_mode == "zh-TW" else (
        "Dùng tiếng Việt, giữ nguyên thuật ngữ kỹ thuật bằng tiếng Trung/Anh."
    )
    if not topic:
        topic = random.choice(QUIZ_TOPICS)
    prompt = (
        f'Tạo 1 câu hỏi trắc nghiệm iPAS AI應用規劃師 về: "{topic}".\n'
        f'Trả về JSON, không thêm gì khác:\n'
        f'{{"q":"câu hỏi","a":"A. ...","b":"B. ...","c":"C. ...","d":"D. ...",'
        f'"ans":"A","explain":"giải thích ngắn"}}\n{lang_instr}'
    )
    try:
        raw = await _groq_chat(
            "You are a quiz generator. Return only valid JSON, no extra text.",
            prompt,
            max_tokens=512,
            temperature=0.9,
        )
        json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if not json_match:
            return "❌ Không tạo được câu hỏi. Dùng /quiz để thử lại."
        data = json.loads(json_match.group())
        _quiz_state[user_id] = data
        return (
            f"📝 <b>Quiz — iPAS AI應用規劃師</b>\n\n"
            f"<b>{data['q']}</b>\n\n"
            f"{data['a']}\n{data['b']}\n{data['c']}\n{data['d']}\n\n"
            f"<i>Trả lời A, B, C hoặc D</i>"
        )
    except Exception as e:
        logger.error(f"Quiz generate error: {e}")
        return f"❌ Lỗi tạo câu hỏi: {e}"


def check_quiz_answer(user_id: int, answer: str) -> Optional[str]:
    state = _quiz_state.get(user_id)
    if not state:
        return None
    correct = state["ans"].upper().strip()[:1]
    user_ans = answer.upper().strip()[:1]
    if user_ans not in "ABCD":
        return None
    if user_ans == correct:
        result = f"✅ <b>Chính xác!</b> Đáp án: <b>{correct}</b>\n\n💡 {state.get('explain', '')}"
    else:
        result = f"❌ <b>Sai!</b> Đáp án đúng: <b>{correct}</b>\n\n💡 {state.get('explain', '')}"
    del _quiz_state[user_id]
    result += "\n\n<i>Dùng /quiz để câu tiếp</i>"
    return result


def has_active_quiz(user_id: int) -> bool:
    return user_id in _quiz_state
