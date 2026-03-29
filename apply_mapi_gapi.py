#!/usr/bin/env python3
"""
apply_mapi_gapi.py — tự chèn /mapi /gapi vào bot.py và discord_bot.py
Run: python apply_mapi_gapi.py
"""
from pathlib import Path

def patch(path, old, new, label):
    p = Path(path)
    src = p.read_text(encoding="utf-8")
    if old not in src:
        print(f"  [SKIP] {label}")
        return False
    p.write_text(src.replace(old, new, 1), encoding="utf-8")
    print(f"  [OK]   {label}")
    return True

# ══════════════════════════════════════════════════════
# bot.py patches
# ══════════════════════════════════════════════════════

# 1. Import
patch("bot.py",
    old='from command_handler import (',
    new='from api_dashboard import cmd_mapi_telegram, cmd_gapi_telegram\nfrom command_handler import (',
    label="bot.py: import api_dashboard"
)

# 2. Register handlers — chèn sau handler /tokens
patch("bot.py",
    old='application.add_handler(CommandHandler("tokens", cmd_tokens))',
    new='application.add_handler(CommandHandler("tokens", cmd_tokens))\n    application.add_handler(CommandHandler("mapi", cmd_mapi_telegram))\n    application.add_handler(CommandHandler("gapi", cmd_gapi_telegram))',
    label="bot.py: register /mapi /gapi handlers"
)

# 3. BotCommand menu
patch("bot.py",
    old='BotCommand("tokens", "Thống kê token và chi phí"),',
    new='BotCommand("tokens", "Thống kê token và chi phí"),\n        BotCommand("mapi", "Mistral AI dashboard"),\n        BotCommand("gapi", "Groq Cloud dashboard"),',
    label="bot.py: add BotCommand entries"
)

# ══════════════════════════════════════════════════════
# discord_bot.py patches
# ══════════════════════════════════════════════════════

# 1. Import
patch("discord_bot.py",
    old='from prompts import MODEL_REGISTRY',
    new='from api_dashboard import cmd_mapi_discord, cmd_gapi_discord\nfrom prompts import MODEL_REGISTRY',
    label="discord_bot.py: import api_dashboard"
)

# 2. Slash commands — chèn trước entry point
patch("discord_bot.py",
    old='# ── FIX 3: Entry point',
    new='''@tree.command(name="mapi", description="Mistral AI dashboard")
async def _mapi(interaction: discord.Interaction):
    await cmd_mapi_discord(interaction)

@tree.command(name="gapi", description="Groq Cloud dashboard")
async def _gapi(interaction: discord.Interaction):
    await cmd_gapi_discord(interaction)

# ── FIX 3: Entry point''',
    label="discord_bot.py: add /mapi /gapi slash commands"
)

print("\n✅ Done. Kiểm tra rồi git add + push.")
