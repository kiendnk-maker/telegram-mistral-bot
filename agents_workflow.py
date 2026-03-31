"""
agents_workflow.py - Multi-agent workflows powered by Groq
"""
import os
import logging
from groq import AsyncGroq
from database import get_setting as _get_setting

logger = logging.getLogger(__name__)

_groq_client = None


def _get_client() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


MODEL_FAST  = "llama-3.1-8b-instant"      # plan, summarize — ultra fast
MODEL_LARGE = "llama-3.3-70b-versatile"   # reasoning, review
MODEL_CODE  = "qwen/qwen3-32b"              # coding, complex tasks — thinking model


async def _lang_instr(user_id: int) -> str:
    """Return language instruction for system prompts based on user's lang_mode setting."""
    lang_mode = await _get_setting(user_id, "lang_mode", "vi")
    if lang_mode == "zh-TW":
        return "必須用繁體中文回覆所有內容。"
    return "Trả lời bằng tiếng Việt."


async def _chat(model: str, system: str, user: str, max_tokens: int = 1024) -> str:
    """Single-turn chat with given Groq model."""
    resp = await _get_client().chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return resp.choices[0].message.content or ""


async def run_multi_agent_workflow(user_id: int, task: str) -> str:
    """
    3-agent pipeline:
    - Agent 1 (fast):  Plan the approach
    - Agent 2 (code):  Execute / implement
    - Agent 3 (large): Review and synthesize
    """
    try:
        lang = await _lang_instr(user_id)
        plan = await _chat(
            MODEL_FAST,
            f"Bạn là chuyên gia lên kế hoạch. Đưa ra kế hoạch rõ ràng, ngắn gọn để giải quyết nhiệm vụ. {lang}",
            f"Nhiệm vụ: {task}\n\nHãy lên kế hoạch chi tiết.",
            max_tokens=512,
        )

        execution = await _chat(
            MODEL_CODE,
            f"Bạn là chuyên gia thực thi. Dựa trên kế hoạch, hãy triển khai chi tiết. Ưu tiên code chất lượng cao. {lang}",
            f"Nhiệm vụ gốc: {task}\n\nKế hoạch:\n{plan}\n\nHãy thực thi chi tiết.",
            max_tokens=1024,
        )

        review = await _chat(
            MODEL_LARGE,
            f"Bạn là chuyên gia phân tích. Đánh giá kết quả và tổng hợp câu trả lời hoàn chỉnh. {lang}",
            f"Nhiệm vụ: {task}\n\nKế hoạch:\n{plan}\n\nThực thi:\n{execution}\n\nHãy tổng hợp và trả lời cuối cùng.",
            max_tokens=1024,
        )

        return (
            f"🤖 <b>Multi-Agent Workflow</b>\n\n"
            f"<b>📋 Kế hoạch:</b>\n{plan}\n\n"
            f"<b>⚙️ Thực thi:</b>\n{execution}\n\n"
            f"<b>✅ Tổng hợp:</b>\n{review}"
        )

    except Exception as e:
        logger.error(f"Multi-agent workflow error: {e}")
        return f"❌ Lỗi trong multi-agent workflow: {str(e)}"


async def run_pro_workflow(user_id: int, task: str) -> str:
    """
    2-step deep workflow using Groq models.
    """
    try:
        lang = await _lang_instr(user_id)
        reasoning = await _chat(
            MODEL_CODE,
            f"Bạn là chuyên gia phân tích sâu. Hãy suy nghĩ kỹ lưỡng về vấn đề, xem xét nhiều góc độ, phân tích ưu nhược điểm. {lang}",
            f"Phân tích sâu về: {task}",
            max_tokens=1500,
        )

        synthesis = await _chat(
            MODEL_LARGE,
            f"Bạn là chuyên gia tổng hợp. Từ phân tích chi tiết, tạo ra câu trả lời súc tích, rõ ràng và hữu ích nhất. {lang}",
            f"Câu hỏi: {task}\n\nPhân tích:\n{reasoning}\n\nHãy tổng hợp câu trả lời ngắn gọn, súc tích.",
            max_tokens=1024,
        )

        return (
            f"🔬 <b>Pro Workflow</b>\n\n"
            f"<b>🧠 Phân tích:</b>\n{reasoning}\n\n"
            f"<b>✅ Kết luận:</b>\n{synthesis}"
        )

    except Exception as e:
        logger.error(f"Pro workflow error: {e}")
        return f"❌ Lỗi trong pro workflow: {str(e)}"


async def run_agentic_loop(user_id: int, task: str) -> str:
    """
    Autonomous agentic loop (up to 5 iterations).
    """
    try:
        lang = await _lang_instr(user_id)
        context = task
        result_history = []
        max_iters = 5

        system_prompt = (
            "Bạn là agent tự động. Mỗi lần nhận nhiệm vụ, hãy:\n"
            "1. Đánh giá tình trạng hiện tại\n"
            "2. Thực hiện bước tiếp theo\n"
            "3. Kết thúc bằng một trong hai:\n"
            "   - CONTINUE: <lý do cần tiếp tục>\n"
            "   - DONE: <kết quả cuối cùng>\n"
            f"{lang}"
        )

        for iteration in range(1, max_iters + 1):
            prompt = (
                f"Nhiệm vụ: {task}\n\n"
                f"Lịch sử ({iteration - 1} bước):\n"
                + ("\n".join(result_history) if result_history else "(Chưa có)")
                + f"\n\nBước {iteration}: Thực hiện và đánh giá."
            )

            step_result = await _chat(MODEL_LARGE, system_prompt, prompt, max_tokens=800)
            result_history.append(f"Bước {iteration}: {step_result}")

            if "DONE:" in step_result.upper():
                done_idx = step_result.upper().find("DONE:")
                final = step_result[done_idx + 5:].strip()
                steps_summary = "\n\n".join(result_history[:-1]) if len(result_history) > 1 else ""
                output = f"🤖 <b>Agentic Loop</b> ({iteration} bước)\n\n"
                if steps_summary:
                    output += f"<b>Quá trình:</b>\n{steps_summary}\n\n"
                output += f"<b>✅ Kết quả:</b>\n{final}"
                return output

        summary = await _chat(
            MODEL_FAST,
            f"Tóm tắt kết quả từ các bước thực hiện. {lang}",
            f"Nhiệm vụ: {task}\n\nCác bước:\n" + "\n\n".join(result_history),
            max_tokens=512,
        )
        return (
            f"🤖 <b>Agentic Loop</b> (đã đạt giới hạn {max_iters} bước)\n\n"
            f"<b>✅ Tóm tắt:</b>\n{summary}"
        )

    except Exception as e:
        logger.error(f"Agentic loop error: {e}")
        return f"❌ Lỗi trong agentic loop: {str(e)}"


async def run_coder_workflow(user_id: int, task: str) -> str:
    """
    Coding workflow: Design (large) → Implement (code) → Review (code).
    """
    try:
        lang = await _lang_instr(user_id)
        design = await _chat(
            MODEL_LARGE,
            f"Bạn là software architect. Phân tích yêu cầu và thiết kế giải pháp. Mô tả cấu trúc, các hàm cần thiết, edge cases cần xử lý. {lang}",
            f"Yêu cầu: {task}\n\nHãy thiết kế giải pháp.",
            max_tokens=512,
        )

        code = await _chat(
            MODEL_CODE,
            f"Bạn là senior developer. Viết code chất lượng cao dựa trên thiết kế. Code phải có: docstring, type hints, error handling, comments. Dùng HTML Telegram để format code. {lang}",
            f"Yêu cầu: {task}\n\nThiết kế:\n{design}\n\nViết code hoàn chỉnh.",
            max_tokens=1500,
        )

        review = await _chat(
            MODEL_CODE,
            f"Bạn là code reviewer. Review code và đề xuất cải thiện: performance, security, readability, missing edge cases. {lang}",
            f"Code:\n{code}\n\nHãy review và nêu điểm cần cải thiện (nếu có).",
            max_tokens=512,
        )

        return (
            f"💻 <b>Coder Workflow</b>\n\n"
            f"<b>🏗 Thiết kế:</b>\n{design}\n\n"
            f"<b>⚙️ Code:</b>\n{code}\n\n"
            f"<b>🔍 Review:</b>\n{review}"
        )

    except Exception as e:
        logger.error(f"Coder workflow error: {e}")
        return f"❌ Lỗi trong coder workflow: {str(e)}"
