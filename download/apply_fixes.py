import os

BASE_DIR = os.path.expanduser("~/telegram-mistral-bot")

def apply_patch():
    llm_path = os.path.join(BASE_DIR, "llm_core.py")
    if os.path.exists(llm_path):
        with open(llm_path, "r", encoding="utf-8") as f:
            code = f.read()
        
        # Chuyển đổi toàn diện SDK sang Mistral bằng string.replace
        code = code.replace("from openai import AsyncOpenAI", "from mistralai import Mistral")
        code = code.replace("client = AsyncOpenAI(api_key=API_KEY)", "client = Mistral(api_key=MISTRAL_API_KEY)")
        code = code.replace("model='mistral-medium'", "model='mistral-large-latest'")
        
        with open(llm_path, "w", encoding="utf-8") as f:
            f.write(code)
        print("✅ Đã vá llm_core.py")

    rag_path = os.path.join(BASE_DIR, "rag_core.py")
    if os.path.exists(rag_path):
        with open(rag_path, "r", encoding="utf-8") as f:
            code = f.read()
        
        # Chèn PyMuPDF fallback nếu chưa có
        if "import fitz" not in code:
            code = "import fitz  # PyMuPDF\n" + code
            
        with open(rag_path, "w", encoding="utf-8") as f:
            f.write(code)
        print("✅ Đã vá rag_core.py")

if __name__ == "__main__":
    apply_patch()
