"""
api_dashboard.py — /mapi and /gapi commands

Telegram: InlineKeyboardButton với url= → mở browser
Discord:  discord.ui.View với url button → mở browser
"""

# ── Telegram handlers ─────────────────────────────────────────────────────────

async def cmd_mapi_telegram(update, context):
    """/mapi — Mistral AI usage dashboard"""
    from telegram import InlineKeyboardMarkup, InlineKeyboardButton
    from telegram.constants import ParseMode

    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("📊 Usage & Cost", url="https://admin.mistral.ai/organization/usage"),
        InlineKeyboardButton("🔑 API Keys",     url="https://console.mistral.ai/api-keys"),
    ], [
        InlineKeyboardButton("💳 Billing",      url="https://admin.mistral.ai/organization/billing"),
        InlineKeyboardButton("⚡ Rate Limits",  url="https://admin.mistral.ai/plateforme/limits"),
    ], [
        InlineKeyboardButton("📈 Models & Pricing", url="https://mistral.ai/pricing"),
    ]])

    await update.message.reply_html(
        "🟧 <b>Mistral AI Dashboard</b>\n\n"
        "Chọn mục bạn muốn xem:",
        reply_markup=kb,
    )


async def cmd_gapi_telegram(update, context):
    """/gapi — Groq Cloud usage dashboard"""
    from telegram import InlineKeyboardMarkup, InlineKeyboardButton

    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("📊 Usage",        url="https://console.groq.com/dashboard/usage"),
        InlineKeyboardButton("🔑 API Keys",     url="https://console.groq.com/keys"),
    ], [
        InlineKeyboardButton("💳 Billing",      url="https://console.groq.com/settings/billing"),
        InlineKeyboardButton("⚡ Rate Limits",  url="https://console.groq.com/settings/limits"),
    ], [
        InlineKeyboardButton("📈 Models & Pricing", url="https://console.groq.com/docs/models"),
    ]])

    await update.message.reply_html(
        "🟢 <b>Groq Cloud Dashboard</b>\n\n"
        "Chọn mục bạn muốn xem:",
        reply_markup=kb,
    )


# ── Discord Views ─────────────────────────────────────────────────────────────

try:
    import discord

    class MistralDashboardView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=None)
            links = [
                ("📊 Usage & Cost",      "https://admin.mistral.ai/organization/usage"),
                ("🔑 API Keys",          "https://console.mistral.ai/api-keys"),
                ("💳 Billing",           "https://admin.mistral.ai/organization/billing"),
                ("⚡ Rate Limits",       "https://admin.mistral.ai/plateforme/limits"),
                ("📈 Models & Pricing",  "https://mistral.ai/pricing"),
            ]
            for label, url in links:
                self.add_item(discord.ui.Button(label=label, url=url, style=discord.ButtonStyle.link))

    class GroqDashboardView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=None)
            links = [
                ("📊 Usage",             "https://console.groq.com/dashboard/usage"),
                ("🔑 API Keys",          "https://console.groq.com/keys"),
                ("💳 Billing",           "https://console.groq.com/settings/billing"),
                ("⚡ Rate Limits",       "https://console.groq.com/settings/limits"),
                ("📈 Models & Pricing",  "https://console.groq.com/docs/models"),
            ]
            for label, url in links:
                self.add_item(discord.ui.Button(label=label, url=url, style=discord.ButtonStyle.link))

    # ── Discord slash command handlers ────────────────────────────────────────────

    async def cmd_mapi_discord(interaction: discord.Interaction):
        await interaction.response.send_message(
            "🟧 **Mistral AI Dashboard**\n\nChọn mục bạn muốn xem:",
            view=MistralDashboardView(),
            ephemeral=True,
        )

    async def cmd_gapi_discord(interaction: discord.Interaction):
        await interaction.response.send_message(
            "🟢 **Groq Cloud Dashboard**\n\nChọn mục bạn muốn xem:",
            view=GroqDashboardView(),
            ephemeral=True,
        )

except ImportError:
    discord = None
