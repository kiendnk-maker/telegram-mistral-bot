#!/bin/bash
# ══════════════════════════════════════════════════════════════
# fix-12bugs.sh — Patch & deploy UltraBot Discord
# Paste this entire script into Termux
# ══════════════════════════════════════════════════════════════
set -e

cd ~/telegram-mistral-bot || { echo "❌ cd failed"; exit 1; }
git pull origin master

echo "🔧 Applying 12-bug fix patch..."

# ── 1. discord_bot.py fixes ──────────────────────────────────

# BUG 4: RetryView model keys → Gemini
sed -i 's/("small", "🔹 Mistral S"),/("flash", "⚡ Flash"),/' discord_bot.py
sed -i 's/("large", "🔵 Mistral L"),/("flash_lite", "💨 Flash Lite"),/' discord_bot.py
sed -i 's/("qwen3", "🌟 Qwen3"),/("flash_think", "💭 2.5 Flash"),/' discord_bot.py
sed -i 's/("gpt_120b", "🧠 GPT 120B"),/("pro", "🧠 2.5 Pro"),/' discord_bot.py
sed -i '/("groq_large", "🦙 Llama 70B"),/d' discord_bot.py

# BUG 5: VisionFollowupView model keys → Gemini
sed -i 's/("llama4", "👁 Vision"),/("flash", "⚡ Flash"),/' discord_bot.py
sed -i 's/("groq_large", "🦙 Llama 70B"),/("flash_think", "💭 2.5 Flash"),/' discord_bot.py
sed -i 's/("gpt_120b", "🧠 GPT 120B"),/("pro", "🧠 2.5 Pro"),/' discord_bot.py
sed -i '/("qwen3", "🌟 Qwen3"),/d' discord_bot.py
sed -i '/("kimi", "🌙 Kimi K2"),/d' discord_bot.py
sed -i 's/current_key: str = "llama4"/current_key: str = "flash"/' discord_bot.py

# BUG 6: Default model_key groq_large → flash
sed -i 's/"model_key", "groq_large"/"model_key", "flash"/g' discord_bot.py

# BUG remaining llama4 default
sed -i 's/"vision_model", "llama4"/"vision_model", "flash"/' discord_bot.py

# BUG 1: add_allowed_user missing added_by + list display fix
sed -i 's/await add_allowed_user(uid)/await add_allowed_user(uid, interaction.user.id)/' discord_bot.py
sed -i "s/join(f\"\`{u}\`\" for u in users)/join(f\"\`{u['user_id']}\`\" for u in users)/" discord_bot.py

# BUG 11: Add gemapi import
sed -i 's/from api_dashboard import cmd_mapi_discord, cmd_gapi_discord/from api_dashboard import cmd_mapi_discord, cmd_gapi_discord, cmd_gemapi_discord/' discord_bot.py

# BUG 2+3: Rewrite focus_tracker command handlers
python3 << 'PYFIX1'
import re

with open("discord_bot.py", "r") as f:
    src = f.read()

# Fix /todo — add_task returns int, not str
old_todo = '''    await interaction.response.send_message(_fmt(await add_task(interaction.user.id, task)), ephemeral=True)'''
new_todo = '''    task_id = await add_task(interaction.user.id, task)
    await interaction.response.send_message(f"✅ Đã thêm task `[{task_id}]`: {task}", ephemeral=True)'''
src = src.replace(old_todo, new_todo)

# Fix /done — complete_task returns bool
old_done = '''    await interaction.response.send_message(_fmt(await complete_task(interaction.user.id, task_id)), ephemeral=True)'''
new_done = '''    ok = await complete_task(interaction.user.id, task_id)
    msg = f"✅ Đã hoàn thành task `[{task_id}]`." if ok else f"❌ Không tìm thấy task `[{task_id}]`."
    await interaction.response.send_message(msg, ephemeral=True)'''
src = src.replace(old_done, new_done)

# Fix /deltask — delete_task returns bool
old_del = '''    await interaction.response.send_message(_fmt(await delete_task(interaction.user.id, task_id)), ephemeral=True)'''
new_del = '''    ok = await delete_task(interaction.user.id, task_id)
    msg = f"🗑 Đã xóa task `[{task_id}]`." if ok else f"❌ Không tìm thấy task `[{task_id}]`."
    await interaction.response.send_message(msg, ephemeral=True)'''
src = src.replace(old_del, new_del)

# Fix /pomodoro — start_pomodoro returns int
old_pom = '''    await interaction.response.send_message(_fmt(await start_pomodoro(interaction.user.id)), ephemeral=True)'''
new_pom = '''    session_id = await start_pomodoro(interaction.user.id)
    count = await get_pomodoro_count_today(interaction.user.id)
    await interaction.response.send_message(
        f"🍅 Pomodoro #{count} bắt đầu! Tập trung 25 phút nhé.", ephemeral=True)'''
src = src.replace(old_pom, new_pom)

# Fix /motivation — get_motivation() is sync, no args
old_mot = '''    await interaction.response.send_message(_fmt(await get_motivation(interaction.user.id)))'''
new_mot = '''    await interaction.response.send_message(_fmt(get_motivation()))'''
src = src.replace(old_mot, new_mot)

# Add /gemapi command after /gapi
old_gapi = '''@tree.command(name="gapi", description="Groq Cloud dashboard")
async def _gapi(interaction: discord.Interaction):
    await cmd_gapi_discord(interaction)

# ── FIX 3'''
new_gapi = '''@tree.command(name="gapi", description="Groq Cloud dashboard")
async def _gapi(interaction: discord.Interaction):
    await cmd_gapi_discord(interaction)

@tree.command(name="gemapi", description="Google Gemini dashboard")
async def _gemapi(interaction: discord.Interaction):
    await cmd_gemapi_discord(interaction)

# ── FIX 3'''
src = src.replace(old_gapi, new_gapi)

with open("discord_bot.py", "w") as f:
    f.write(src)

print("  ✅ discord_bot.py patched")
PYFIX1

# ── 2. rag_core.py — migrate Mistral embeddings → Gemini ─────

python3 << 'PYFIX2'
with open("rag_core.py", "r") as f:
    src = f.read()

# Remove dead fitz import
src = src.replace('import fitz  # PyMuPDF\n', '')

# Update docstring
src = src.replace('rag_core.py - ChromaDB RAG system with Mistral embeddings',
                  'rag_core.py - ChromaDB RAG system with Gemini embeddings')

# Add google genai import after chromadb
src = src.replace('import chromadb\n',
                  'import chromadb\nfrom google import genai\n')

# Replace entire embedding section
old_embed = '''_embed_client = None

def _get_embed_client():
    global _embed_client
    if _embed_client is None:
        from mistralai.client import MistralClient
        _embed_client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
    return _embed_client


async def _get_embedding(text: str) -> list[float]:
    """Get embedding from Mistral API (reuses singleton client)."""
    def _embed():
        response = _get_embed_client().embeddings(
            model="mistral-embed",
            input=[text]
        )
        return response.data[0].embedding

    return await asyncio.get_event_loop().run_in_executor(None, _embed)'''

new_embed = '''_gemini_client: genai.Client | None = None


def _get_gemini() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _gemini_client


async def _get_embedding(text: str) -> list[float]:
    """Get embedding from Gemini API."""
    def _embed():
        response = _get_gemini().models.embed_content(
            model="gemini-embedding-001",
            contents=text,
        )
        return response.embeddings[0].values

    return await asyncio.to_thread(_embed)'''

src = src.replace(old_embed, new_embed)

# Fix remaining deprecated asyncio call
src = src.replace(
    'await asyncio.get_event_loop().run_in_executor(None, _extract)',
    'await asyncio.to_thread(_extract)')

with open("rag_core.py", "w") as f:
    f.write(src)

print("  ✅ rag_core.py patched")
PYFIX2

# ── 3. tracker_core.py — add Gemini pricing ──────────────────

python3 << 'PYFIX3'
with open("tracker_core.py", "r") as f:
    src = f.read()

old_pricing = '''# Pricing per 1M tokens in USD
PRICING = {
    # Mistral
    "small":      {"input": 0.2,    "output": 0.6},'''

new_pricing = '''# Pricing per 1M tokens in USD
PRICING = {
    # Gemini (current)
    "flash":       {"input": 0.10,   "output": 0.40},
    "flash_lite":  {"input": 0.0,    "output": 0.0},
    "flash_think": {"input": 0.15,   "output": 0.60},
    "pro":         {"input": 1.25,   "output": 5.00},
    "vision":      {"input": 0.10,   "output": 0.40},
    # Legacy (kept for old token_usage rows)
    "small":      {"input": 0.2,    "output": 0.6},'''

src = src.replace(old_pricing, new_pricing)

# Fix vision pricing line (was 3.0/9.0, now under Gemini above)
src = src.replace('    "vision":     {"input": 3.0,    "output": 9.0},\n    # Groq\n',
                  '    # Groq\n')

# Add Gemini to MODEL_DISPLAY
old_display = '''MODEL_DISPLAY = {
    "small":      "Mistral Small ⚡",'''

new_display = '''MODEL_DISPLAY = {
    # Gemini (current)
    "flash":       "Gemini Flash ⚡",
    "flash_lite":  "Gemini Flash Lite 💨",
    "flash_think": "Gemini 2.5 Flash 💭",
    "pro":         "Gemini 2.5 Pro 🧠",
    "vision":      "Gemini Vision 👁",
    # Legacy
    "small":      "Mistral Small ⚡",'''

src = src.replace(old_display, new_display)

# Fix stale Pixtral
src = src.replace('"vision":     "Pixtral Large 👁",\n', '')

with open("tracker_core.py", "w") as f:
    f.write(src)

print("  ✅ tracker_core.py patched")
PYFIX3

# ── 4. api_dashboard.py — add Gemini dashboard ───────────────

python3 << 'PYFIX4'
with open("api_dashboard.py", "r") as f:
    src = f.read()

gemini_view = '''
    class GeminiDashboardView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=None)
            links = [
                ("📊 Usage",             "https://aistudio.google.com/apikey"),
                ("🔑 API Keys",          "https://aistudio.google.com/apikey"),
                ("💳 Billing",           "https://console.cloud.google.com/billing"),
                ("📈 Models & Pricing",  "https://ai.google.dev/gemini-api/docs/models/gemini"),
            ]
            for label, url in links:
                self.add_item(discord.ui.Button(label=label, url=url, style=discord.ButtonStyle.link))

    class GroqDashboardView'''

src = src.replace('    class GroqDashboardView', gemini_view)

gemapi_handler = '''
    async def cmd_gemapi_discord(interaction: discord.Interaction):
        await interaction.response.send_message(
            "🔵 **Google Gemini Dashboard**\\n\\nChọn mục bạn muốn xem:",
            view=GeminiDashboardView(),
            ephemeral=True,
        )

except ImportError:'''

src = src.replace('except ImportError:', gemapi_handler)

with open("api_dashboard.py", "w") as f:
    f.write(src)

print("  ✅ api_dashboard.py patched")
PYFIX4

# ── 5. Syntax check ──────────────────────────────────────────

echo ""
echo "🔍 Syntax check..."
python3 -c "
import ast
for f in ['discord_bot.py','rag_core.py','tracker_core.py','api_dashboard.py']:
    with open(f) as fh: ast.parse(fh.read())
    print(f'  ✅ {f}')
print('All OK')
"

# ── 6. Git commit & push → Railway auto-deploy ───────────────

echo ""
echo "📦 Committing & pushing..."
git add -A
git commit -m "fix: 12-bug patch — Gemini migration cleanup

- BUG 1:  add_allowed_user() missing added_by arg
- BUG 2:  get_motivation() called as async with wrong args
- BUG 3:  _fmt() receives bool/int instead of str (todo/done/deltask/pomodoro)
- BUG 4:  RetryView references non-existent Mistral/Groq model keys
- BUG 5:  VisionFollowupView references non-existent model keys
- BUG 6:  Default model_key 'groq_large' → 'flash'
- BUG 7:  tracker_core.py missing Gemini pricing (cost always \$0)
- BUG 8:  rag_core.py deprecated mistralai v0.x SDK → Gemini embeddings
- BUG 9:  rag_core.py dead 'import fitz' removed
- BUG 10: asyncio.get_event_loop() → asyncio.to_thread() (Python 3.12)
- BUG 11: Added /gemapi command (Google Gemini dashboard)
- BUG 12: mistralai dependency removed from rag_core.py"

git push origin master

echo ""
echo "══════════════════════════════════════════════════"
echo "✅ Done! 12 bugs fixed. Railway will auto-deploy."
echo "══════════════════════════════════════════════════"
