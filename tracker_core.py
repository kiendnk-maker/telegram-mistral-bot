"""
tracker_core.py - Token usage tracker with cost estimation
"""
from database import get_token_report

# Pricing per 1M tokens in USD
PRICING = {
    # Mistral
    "small":      {"input": 0.2,   "output": 0.6},
    "large":      {"input": 2.0,   "output": 6.0},
    "codestral":  {"input": 0.3,   "output": 0.9},
    "vision":     {"input": 3.0,   "output": 9.0},
    # Groq (free tier / on-demand pricing approximation)
    "groq_fast":  {"input": 0.05,  "output": 0.08},
    "groq_large": {"input": 0.59,  "output": 0.79},
    "llama4":     {"input": 0.11,  "output": 0.34},
    "qwen3":      {"input": 0.29,  "output": 0.59},
    "kimi":       {"input": 1.0,   "output": 3.0},
}

MODEL_DISPLAY = {
    "small":      "Mistral Small ⚡",
    "large":      "Mistral Large 🧠",
    "codestral":  "Codestral 💻",
    "vision":     "Pixtral Large 👁",
    "groq_fast":  "Llama 3.1 8B ⚡ (Groq)",
    "groq_large": "Llama 3.3 70B 🦙 (Groq)",
    "llama4":     "Llama 4 Scout 🚀 (Groq)",
    "qwen3":      "Qwen3 32B 🌟 (Groq)",
    "kimi":       "Kimi K2 🌙 (Groq)",
}


def _calc_cost(model_key: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = PRICING.get(model_key, {"input": 0.0, "output": 0.0})
    cost = (prompt_tokens / 1_000_000) * pricing["input"]
    cost += (completion_tokens / 1_000_000) * pricing["output"]
    return cost


async def get_usage_report(user_id: int) -> str:
    """Returns formatted HTML string with usage stats."""
    rows = await get_token_report(user_id)
    if not rows:
        return "📊 <b>Chưa có dữ liệu sử dụng.</b>"

    total_cost = 0.0
    total_prompt = 0
    total_completion = 0
    total_calls = 0

    lines = ["📊 <b>Thống kê token của bạn:</b>\n"]

    for row in rows:
        model_key = row["model"]
        prompt_t = row["total_prompt"]
        completion_t = row["total_completion"]
        calls = row["call_count"]
        cost = _calc_cost(model_key, prompt_t, completion_t)

        total_cost += cost
        total_prompt += prompt_t
        total_completion += completion_t
        total_calls += calls

        display_name = MODEL_DISPLAY.get(model_key, model_key)
        lines.append(
            f"<b>{display_name}</b>\n"
            f"  Số lượt: <code>{calls}</code>\n"
            f"  Input: <code>{prompt_t:,}</code> tokens\n"
            f"  Output: <code>{completion_t:,}</code> tokens\n"
            f"  Chi phí: <code>${cost:.4f}</code>\n"
        )

    lines.append(
        f"\n<b>Tổng cộng:</b>\n"
        f"  Lượt gọi: <code>{total_calls}</code>\n"
        f"  Tokens: <code>{total_prompt + total_completion:,}</code>\n"
        f"  Ước tính: <code>${total_cost:.4f} USD</code>"
    )

    return "\n".join(lines)
