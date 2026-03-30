"""
google_services.py — Google Calendar, Gmail, Drive via OAuth2 + httpx
"""
import os, json, re, logging
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlencode
import httpx, aiosqlite
from database import DB_PATH

logger = logging.getLogger(__name__)
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
REDIRECT_URI = os.getenv("PUBLIC_URL", "http://localhost").rstrip("/") + "/oauth/callback"
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

# ── Token DB ─────────────────────────────────────────────────────────────────
async def _init_google_table():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""CREATE TABLE IF NOT EXISTS google_tokens (
            user_id INTEGER PRIMARY KEY, access_token TEXT NOT NULL,
            refresh_token TEXT NOT NULL, expires_at TEXT NOT NULL, email TEXT)""")
        await db.commit()

async def _save_tokens(user_id, access_token, refresh_token, expires_in, email=""):
    expires_at = (datetime.utcnow() + timedelta(seconds=expires_in)).isoformat()
    await _init_google_table()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO google_tokens VALUES (?,?,?,?,?)",
            (user_id, access_token, refresh_token, expires_at, email))
        await db.commit()

async def _get_tokens(user_id):
    await _init_google_table()
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM google_tokens WHERE user_id=?", (user_id,)) as c:
            row = await c.fetchone()
    return dict(row) if row else None

async def _get_valid_token(user_id):
    tokens = await _get_tokens(user_id)
    if not tokens: return None
    if datetime.utcnow() < datetime.fromisoformat(tokens["expires_at"]) - timedelta(minutes=5):
        return tokens["access_token"]
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post("https://oauth2.googleapis.com/token", data={
                "client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET,
                "refresh_token": tokens["refresh_token"], "grant_type": "refresh_token"})
            d = r.json()
        if "access_token" not in d: return None
        await _save_tokens(user_id, d["access_token"], tokens["refresh_token"],
                          d.get("expires_in", 3600), tokens.get("email",""))
        return d["access_token"]
    except Exception as e:
        logger.error(f"Token refresh: {e}"); return None

# ── OAuth2 ───────────────────────────────────────────────────────────────────
def get_auth_url(state: str = "") -> str:
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
    }
    if state:
        params["state"] = state
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"

async def exchange_code(user_id, code):
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post("https://oauth2.googleapis.com/token", data={
                "client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET,
                "code": code, "grant_type": "authorization_code", "redirect_uri": REDIRECT_URI})
            d = r.json()
        if "error" in d: return f"❌ {d.get('error_description', d['error'])}"
        email = ""
        try:
            async with httpx.AsyncClient() as c:
                p = await c.get("https://www.googleapis.com/oauth2/v2/userinfo",
                               headers={"Authorization": f"Bearer {d['access_token']}"})
                email = p.json().get("email","")
        except: pass
        await _save_tokens(user_id, d["access_token"], d.get("refresh_token",""),
                          d.get("expires_in",3600), email)
        return f"✅ Đã kết nối Google!\n📧 {email}\n\nDùng /cal, /gmail, /gdrive"
    except Exception as e: return f"❌ Lỗi: {e}"

async def is_connected(user_id):
    return (await _get_tokens(user_id)) is not None

# ── Calendar ─────────────────────────────────────────────────────────────────
async def list_events(user_id, days=7):
    token = await _get_valid_token(user_id)
    if not token: return "❌ Chưa kết nối Google. Dùng /gauth"
    now = datetime.utcnow().isoformat() + "Z"
    end = (datetime.utcnow() + timedelta(days=days)).isoformat() + "Z"
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get("https://www.googleapis.com/calendar/v3/calendars/primary/events",
                headers={"Authorization": f"Bearer {token}"},
                params={"timeMin": now, "timeMax": end, "maxResults": 10,
                        "singleEvents": "true", "orderBy": "startTime"})
            data = r.json()
        events = data.get("items", [])
        if not events: return f"📅 Không có sự kiện nào trong {days} ngày tới."
        lines = [f"📅 <b>Lịch {days} ngày tới ({len(events)}):</b>\n"]
        for ev in events:
            s = ev.get("start",{})
            dt_str = s.get("dateTime", s.get("date",""))
            try:
                dt = datetime.fromisoformat(dt_str.replace("Z","+00:00"))
                ts = dt.strftime("%d/%m %H:%M")
            except: ts = dt_str[:16]
            lines.append(f"• <b>{ts}</b> — {ev.get('summary','(Không tiêu đề)')}")
        return "\n".join(lines)
    except Exception as e: return f"❌ Lỗi Calendar: {e}"

async def add_event(user_id, title, start_iso, end_iso):
    token = await _get_valid_token(user_id)
    if not token: return "❌ Chưa kết nối Google. Dùng /gauth"
    body = {"summary": title,
            "start": {"dateTime": start_iso, "timeZone": "Asia/Taipei"},
            "end": {"dateTime": end_iso, "timeZone": "Asia/Taipei"}}
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post("https://www.googleapis.com/calendar/v3/calendars/primary/events",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=body)
            data = r.json()
        if "error" in data: return f"❌ {data['error'].get('message','')}"
        return f"✅ Đã thêm: <b>{title}</b>\n🔗 {data.get('htmlLink','')}"
    except Exception as e: return f"❌ Lỗi: {e}"

# ── Gmail ────────────────────────────────────────────────────────────────────
async def list_unread(user_id, count=5):
    token = await _get_valid_token(user_id)
    if not token: return "❌ Chưa kết nối Google. Dùng /gauth"
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get("https://gmail.googleapis.com/gmail/v1/users/me/messages",
                headers={"Authorization": f"Bearer {token}"},
                params={"q": "is:unread", "maxResults": count})
            msgs = r.json().get("messages", [])
        if not msgs: return "📧 Không có email chưa đọc! 🎉"
        lines = [f"📧 <b>{len(msgs)} email chưa đọc:</b>\n"]
        async with httpx.AsyncClient() as c:
            for m in msgs[:count]:
                r = await c.get(f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{m['id']}",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"format": "metadata", "metadataHeaders": "Subject,From"})
                d = r.json()
                hdrs = {h["name"]: h["value"] for h in d.get("payload",{}).get("headers",[])}
                subj = hdrs.get("Subject","(Không tiêu đề)")
                sender = hdrs.get("From","")
                if "<" in sender: sender = sender.split("<")[0].strip().strip('"')
                lines.append(f"• <b>{subj}</b>\n  <i>{sender}</i>")
        return "\n\n".join(lines)
    except Exception as e: return f"❌ Lỗi Gmail: {e}"

# ── Drive ────────────────────────────────────────────────────────────────────
async def search_drive(user_id, query, count=5):
    token = await _get_valid_token(user_id)
    if not token: return "❌ Chưa kết nối Google. Dùng /gauth"
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get("https://www.googleapis.com/drive/v3/files",
                headers={"Authorization": f"Bearer {token}"},
                params={"q": f"name contains '{query}' and trashed=false",
                        "pageSize": count, "orderBy": "modifiedTime desc",
                        "fields": "files(id,name,mimeType,webViewLink,modifiedTime)"})
            files = r.json().get("files", [])
        if not files: return f"📁 Không tìm thấy: {query}"
        icons = {"document":"📝","spreadsheet":"📊","presentation":"📽","folder":"📂"}
        lines = [f"📁 <b>Tìm: \"{query}\" ({len(files)} file)</b>\n"]
        for f in files:
            mt = f.get("mimeType","")
            icon = next((v for k,v in icons.items() if k in mt), "📄")
            lines.append(f"{icon} <b>{f['name']}</b> ({f.get('modifiedTime','')[:10]})\n  🔗 {f.get('webViewLink','')}")
        return "\n\n".join(lines)
    except Exception as e: return f"❌ Lỗi Drive: {e}"
