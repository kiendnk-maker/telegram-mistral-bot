"""
prompts.py - System prompts and model registry
"""

BASE_PROMPT = """Bạn là trợ lý AI thông minh tên là Ultra Bolt.
Trả lời bằng tiếng Việt. Định dạng HTML Telegram:
- <b>text</b> cho chữ đậm
- <i>text</i> cho chữ nghiêng
- <code>text</code> cho code ngắn
- <pre>text</pre> cho code block
Dùng emoji tự nhiên. Trả lời ngắn gọn, súc tích."""

REASONING_SUFFIX = "\nHãy suy nghĩ kỹ trước khi trả lời. Chỉ hiển thị câu trả lời cuối cùng."
CODER_SUFFIX = "\nBạn là chuyên gia lập trình. Ưu tiên code chất lượng cao, có comment."
SEARCH_SUFFIX = "\nKhi trả lời, hãy trích dẫn nguồn thông tin nếu có."

MODEL_REGISTRY = {
    "small": {
        "model_id": "mistral-small-latest",
        "name": "Mistral Small ⚡",
        "desc": "Nhanh, phù hợp câu hỏi thông thường",
    },
    "large": {
        "model_id": "mistral-large-latest",
        "name": "Mistral Large 🧠",
        "desc": "Thông minh hơn, phù hợp phân tích sâu",
    },
    "codestral": {
        "model_id": "codestral-latest",
        "name": "Codestral 💻",
        "desc": "Chuyên code",
    },
    "vision": {
        "model_id": "pixtral-large-latest",
        "name": "Pixtral Large 👁",
        "desc": "Xử lý ảnh",
    },
}


def get_system_prompt(model_key: str = "small", profile: str = None) -> str:
    prompt = BASE_PROMPT
    if profile:
        prompt += f"\n\nThông tin người dùng: {profile}"
    if model_key == "codestral":
        prompt += CODER_SUFFIX
    elif model_key in ("large",):
        prompt += REASONING_SUFFIX
    return prompt
