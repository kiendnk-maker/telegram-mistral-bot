import os

BASE_DIR = os.path.expanduser("~/telegram-mistral-bot")
# Đổi tên file đích bên dưới cho đúng với luồng Discord của bạn
TARGET_FILE = os.path.join(BASE_DIR, "discord_core.py") 

def patch_discord():
    if not os.path.exists(TARGET_FILE):
        print("⚠️ Không tìm thấy file đích.")
        return
        
    with open(TARGET_FILE, 'r', encoding='utf-8') as f:
        code = f.read()
        
    # Áp dụng thay thế nguyên khối, không dùng regex
    code = code.replace("OLD_DISCORD_LOGIC_HERE", "NEW_DISCORD_LOGIC_HERE")
    
    with open(TARGET_FILE, 'w', encoding='utf-8') as f:
        f.write(code)
    print("✅ Đã vá mã nguồn Discord thành công.")

if __name__ == "__main__":
    patch_discord()
