"""
oauth_server.py — Mini web server for Google OAuth2 callback
Runs alongside Telegram bot on Railway $PORT
"""
import os
import logging
from aiohttp import web

logger = logging.getLogger(__name__)


async def oauth_callback(request: web.Request) -> web.Response:
    """Handle Google OAuth2 redirect callback."""
    code = request.query.get("code", "")
    state = request.query.get("state", "")  # user_id
    error = request.query.get("error", "")

    if error:
        return web.Response(
            text=f"""<html><body style="background:#1a1a2e;color:#e0e0e0;font-family:sans-serif;text-align:center;padding:60px">
            <h1>❌ Đã hủy kết nối</h1><p>{error}</p>
            <p>Quay lại bot và thử /gauth lại.</p></body></html>""",
            content_type="text/html")

    if not code or not state:
        return web.Response(
            text="""<html><body style="background:#1a1a2e;color:#e0e0e0;font-family:sans-serif;text-align:center;padding:60px">
            <h1>❌ Thiếu thông tin</h1><p>Quay lại bot và dùng /gauth</p></body></html>""",
            content_type="text/html")

    # Exchange code for tokens
    try:
        user_id = int(state)
        from google_services import exchange_code
        result = await exchange_code(user_id, code)

        # Check success
        success = "✅" in result
        if success:
            html = f"""<html><body style="background:#1a1a2e;color:#e0e0e0;font-family:sans-serif;text-align:center;padding:60px">
            <h1>✅ Kết nối thành công!</h1>
            <p style="font-size:18px">Quay lại Telegram bot để dùng:</p>
            <p><code>/cal</code> — Lịch &nbsp; <code>/gmail</code> — Email &nbsp; <code>/gdrive</code> — Drive</p>
            <p style="margin-top:30px;opacity:0.6">Có thể đóng tab này.</p></body></html>"""
        else:
            html = f"""<html><body style="background:#1a1a2e;color:#e0e0e0;font-family:sans-serif;text-align:center;padding:60px">
            <h1>❌ Lỗi kết nối</h1><p>{result}</p>
            <p>Quay lại bot và thử /gauth lại.</p></body></html>"""

        # Also notify user via Telegram if bot reference available
        bot_ref = request.app.get("telegram_bot")
        if bot_ref and success:
            try:
                await bot_ref.send_message(
                    chat_id=user_id,
                    text="✅ Google đã kết nối thành công!\n\nDùng /cal, /gmail, /gdrive",
                )
            except Exception as e:
                logger.warning(f"Cannot notify user {user_id}: {e}")

        return web.Response(text=html, content_type="text/html")

    except Exception as e:
        logger.error(f"OAuth callback error: {e}", exc_info=True)
        return web.Response(
            text=f"""<html><body style="background:#1a1a2e;color:#e0e0e0;font-family:sans-serif;text-align:center;padding:60px">
            <h1>❌ Lỗi hệ thống</h1><p>{e}</p></body></html>""",
            content_type="text/html")


async def health(request: web.Request) -> web.Response:
    """Health check endpoint for Railway."""
    return web.json_response({"status": "ok", "bot": "UltraBolt"})


def create_oauth_app(telegram_bot=None) -> web.Application:
    """Create aiohttp app with OAuth callback route."""
    app = web.Application()
    app["telegram_bot"] = telegram_bot
    app.router.add_get("/oauth/callback", oauth_callback)
    app.router.add_get("/health", health)
    app.router.add_get("/", health)
    return app
