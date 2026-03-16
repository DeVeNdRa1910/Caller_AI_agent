"""
main.py — Multi-Tenant AI Voice Pipeline with RAG

ARCHITECTURE:
  - Multi-tenancy: each tenant has isolated FAISS knowledge base + API key auth
  - RAG: every turn retrieves relevant chunks from tenant's uploaded documents
  - Agent behaviour is document-driven — no hardcoded business knowledge
  - Supports both inbound (Twilio webhook) and outbound (POST /call-user) calls
  - Ultra-low latency: parallel TTS, pre-rendered cache, streaming LLM

FLOW:
  1. POST /admin/tenants          → create tenant, get api_key
  2. POST /tenant/{id}/documents  → upload PDF/TXT knowledge base
  3. POST /call-user?mobile_number=+91...&tenant_id=...  → outbound call
     OR set Twilio webhook → /voice?tenant_id=...        → inbound call
  4. On each turn: RAG retrieve → LLM (grounded) → TTS → Twilio

FIXES IN THIS VERSION:
  1. TTS error on /call-user: opening_text was empty when company_name blank
  2. LLM ignoring knowledge base: model=claude/llama was falling back to
     training data. Fixed by injecting EXPLICIT allowed-languages list parsed
     from retrieved context into the system prompt header.
  3. Memory leak: _call_history / _call_tenant_map now cleaned on all terminal
     Twilio statuses (busy, no-answer, failed, canceled) not just "completed".
  4. History append race: history is only written after full pipeline success.
  5. History size cap: prevents unbounded memory growth on long calls.
  6. Untracked background tasks now log exceptions.
  7. serve_audio no longer deletes on first serve (Twilio retries protection).
"""

import asyncio
import logging
import os
import re
import time
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, WebSocket, Header, HTTPException, UploadFile, File
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field
from groq import AsyncGroq
from twilio.rest import Client

from tts import (
    generate_tts_async,
    synthesise_text,
    llm_to_tts_stream,
    close_http_client,
    warmup_ws,
    AUDIO_DIR,
)
from media_stream import handle_media_stream
from rag_pipeline import (
    rag_startup,
    retrieve_context,
    ingest_file_bytes,
    ingest_text,
    list_tenant_documents,
    delete_tenant_documents,
    delete_tenant_collection,
    build_rag_system_prompt,
)
from tenant_manager import (
    create_tenant,
    get_tenant,
    get_tenant_by_api_key,
    list_tenants,
    update_tenant,
    delete_tenant,
    rotate_api_key,
)

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── App & clients ──────────────────────────────────────────────────────────────

app         = FastAPI(title="Multi-Tenant Voice AI Pipeline")
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

_call_history:    dict[str, list[dict]] = {}
_call_tenant_map: dict[str, str]        = {}
_audio_cache:     dict[str, str]        = {}

_SENTENCE_END     = re.compile(r"[।.?!]\s*")
_MAX_BUFFER_CHARS = 120

# Terminal call statuses — clean up memory on any of these
_TERMINAL_CALL_STATUSES = {"completed", "busy", "no-answer", "failed", "canceled"}

# Cap history to prevent unbounded memory growth
_MAX_HISTORY_TURNS = 20  # = 40 messages (user + assistant pairs)

# ── Pydantic models ────────────────────────────────────────────────────────────

class CreateTenantRequest(BaseModel):
    tenant_id:              str           = Field(...,     description="Unique ID e.g. sharma_logistics")
    name:                   str           = Field(...,     description="Display name e.g. Sharma Logistics")
    agent_name:             str           = Field(...,     description="AI agent name — use Hindi pronunciation spelling e.g. सोनी not SONY")
    language_preference:    str           = Field("hi-IN", description="Default language code")
    system_prompt_override: Optional[str] = Field(None,    description="Custom system prompt — leave blank to use default")
    extra_context:          str           = Field("",      description="Static text always injected")


class UpdateTenantRequest(BaseModel):
    name:                   Optional[str]  = Field(None)
    agent_name:             Optional[str]  = Field(None)
    language_preference:    Optional[str]  = Field(None)
    system_prompt_override: Optional[str]  = Field(None)
    extra_context:          Optional[str]  = Field(None)
    active:                 Optional[bool] = Field(None)


class UploadTextRequest(BaseModel):
    text:        str = Field(...,           description="Raw text to ingest")
    source_name: str = Field("text_upload", description="Label for this document")


# ── Base system prompt ─────────────────────────────────────────────────────────
# This is the CALL BEHAVIOUR section only.
# ALL factual knowledge is injected via RAG in build_rag_system_prompt().
# This prompt is placed AFTER the knowledge base in the final prompt —
# knowledge base always comes first so LLM gives it highest weight.

BASE_SYSTEM_PROMPT = """You are {agent_name}, a sales agent on a LIVE PHONE CALL.
Everything you know is ONLY from the Knowledge Base (KB).

═══ LANGUAGE ═══
Reply in ROMANIZED HINGLISH (English letters, Hindi-English mix).
NEVER output Devanagari characters. Output MUST be Roman script only.

═══ CONVERSATION FLOW ═══
Follow these steps in order. Pick ALL details (company name, service, questions, next steps) from the KB.

STEP 1 — GREETING + LANGUAGE PREFERENCE:
  OUTBOUND: Already done. DO NOT repeat.
  INBOUND: Introduce yourself with your name and company name from KB. Ask language preference.

STEP 2 — STATE REASON + CHECK GOOD TIME:
  Tell the user why you are calling (use the service/product from KB).
  Ask if this is a good time to talk.
  If NO → ask when to call back → confirm the time → end.
  If YES → go to Step 3.

STEP 3 — CONFIRM THE NEED:
  Confirm with the user that they need the service/product described in KB.
  Wait for yes/no.

STEP 4 — COLLECT INFORMATION:
  Collect details the KB says are needed.
  Ask in LOGICAL GROUPS (2-3 related questions per turn).
  Wait for answers before asking the next group.
  Continue until all KB-required info is collected.

STEP 5 — NEXT STEPS:
  Offer whatever next steps the KB describes (quotation, callback, demo, etc.).
  Ask the user's preference on how to receive it.

STEP 6 — SCHEDULE (if KB mentions any scheduling like survey/visit/demo):
  Schedule it. Ask for convenient time and any other details needed.

STEP 7 — WRAP UP:
  Thank them, confirm what happens next, end politely.

═══ RESPONDING TO THE USER ═══
CRITICAL: Always respond to what the user ACTUALLY said FIRST.
- If user asks to repeat / says "phir se bolo" / "dobara bolo" / "kya bola" → REPEAT your last question.
- If user asks a question → ANSWER it first, then continue the flow.
- If user says something unclear → Politely ask them to repeat.
- If user gives a partial answer → Acknowledge it, then ask the remaining part.
- ONLY move to the next step AFTER the user has fully answered the current question.

═══ RULES ═══
- 1-2 short sentences per reply MAX. This is a phone call.
- NEVER echo back what the customer said. Just move forward.
- NEVER re-ask something already answered. Check conversation history.
- NEVER make up info not in KB. Say you will confirm with the team.
- NEVER use bullets, lists, or markdown.
- ALWAYS pronounce {agent_name} as one word.
"""

# ── Fixed responses & opening ──────────────────────────────────────────────────

FIXED_RESPONSES: dict[str, str] = {
    "unclear_hi":   "Sorry, clear nahi sun paayi. Aap dobara bol sakte hai?",
    "unclear_en":   "Sorry, I didn't catch that. Could you please repeat?",
    "unclear_both": "Sorry, clear nahi hua. Aap ek baar aur bol dijiye?",
}

# ── Auth helpers ───────────────────────────────────────────────────────────────

def _require_admin(api_key: str | None):
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY not configured")
    if api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin API key")


async def _require_tenant_auth(tenant_id: str, api_key: str | None) -> dict:
    if not api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required")
    tenant = await get_tenant_by_api_key(api_key)
    if not tenant or tenant["tenant_id"] != tenant_id:
        raise HTTPException(status_code=403, detail="Invalid API key for this tenant")
    return tenant


# ── Tenant helpers ─────────────────────────────────────────────────────────────

async def _get_base_prompt(tenant_id: str | None) -> str:
    if not tenant_id:
        return BASE_SYSTEM_PROMPT.replace("{agent_name}", "AI Assistant")
    tenant = await get_tenant(tenant_id)
    if not tenant:
        return BASE_SYSTEM_PROMPT.replace("{agent_name}", "AI Assistant")

    agent_name = tenant.get("agent_name") or "AI Assistant"

    override = (tenant.get("system_prompt_override") or "").strip()
    _INVALID_OVERRIDES = {"string", "str", "none", "null", "test", ""}
    if override and override.lower() not in _INVALID_OVERRIDES and len(override) > 20:
        prompt = override
    else:
        prompt = BASE_SYSTEM_PROMPT.replace("{agent_name}", agent_name)

    extra = tenant.get("extra_context", "").strip()
    if extra:
        prompt += f"\n\nADDITIONAL CONTEXT:\n{extra}"
    return prompt


# ── Audio cache helpers ────────────────────────────────────────────────────────

def _cache_get(text: str) -> str | None:
    fn = _audio_cache.get(text.strip())
    if fn and os.path.isfile(os.path.join(AUDIO_DIR, fn)):
        return fn
    return None


def _cache_consume(text: str) -> str | None:
    fn = _cache_get(text)
    if fn:
        _audio_cache.pop(text.strip(), None)
        task = asyncio.create_task(_rerender_bg(text))
        # FIX: log exceptions from background tasks so they're not silently swallowed
        task.add_done_callback(
            lambda t: log.warning("Cache re-render error: %s", t.exception())
            if not t.cancelled() and t.exception() else None
        )
    return fn


async def _rerender_bg(text: str):
    try:
        fn = await synthesise_text(text)
        _audio_cache[text.strip()] = fn
    except Exception as e:
        log.warning("Cache re-render failed: %s", e)


async def _prerender_all():
    await asyncio.gather(*[_render_one(k, t) for k, t in FIXED_RESPONSES.items()])


async def _render_one(key: str, text: str):
    try:
        fn = await synthesise_text(text)
        _audio_cache[text.strip()] = fn
        log.info("  ✅ [%s] cached", key)
    except Exception as e:
        log.warning("  ❌ [%s] failed: %s", key, e)


# ── LLM pipeline ──────────────────────────────────────────────────────────────

async def _stream_llm_sentences(messages: list[dict]):
    stream = await groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=128,
        temperature=0.1,
        stream=True,
        tool_choice="none",
    )
    buffer = ""
    async for chunk in stream:
        delta   = (chunk.choices[0].delta.content or "") if chunk.choices else ""
        buffer += delta
        m = _SENTENCE_END.search(buffer)
        if m:
            sentence = buffer[: m.end()].strip()
            buffer   = buffer[m.end():]
            if sentence:
                yield sentence
        elif len(buffer.strip()) >= _MAX_BUFFER_CHARS:
            yield buffer.strip()
            buffer = ""
    if buffer.strip():
        yield buffer.strip()


async def _run_pipeline(
    messages:   list[dict],
    call_sid:   str,
    user_input: str,
) -> tuple[list[str], str]:
    t0        = time.perf_counter()
    sentences : list[str] = []

    async for sentence in _stream_llm_sentences(messages):
        if sentence.strip():
            sentences.append(sentence.strip())

    ai_reply = " ".join(sentences).strip() or "Sorry, clear nahi sun paayi. Aap dobara bol sakte hai?"

    if not sentences:
        stream = await groq_client.chat.completions.create(
            model=GROQ_MODEL, messages=messages,
            max_tokens=128, temperature=0.1, stream=True, tool_choice="none",
        )
        fn, ai_reply = await llm_to_tts_stream(stream)
        ai_reply     = ai_reply.strip() or "Sorry, clear nahi sun paayi. Aap dobara bol sakte hai?"
        filenames    = [fn]
    else:
        tts_tasks = [generate_tts_async(s) for s in sentences]
        filenames = list(await asyncio.gather(*tts_tasks))
        log.info("⏱  TTS: %.3fs | %d segments", time.perf_counter() - t0, len(filenames))

    # FIX: only append to history after full pipeline success, and cap size
    if call_sid:
        hist = _call_history.setdefault(call_sid, [])
        hist.append({"role": "user",      "content": user_input})
        hist.append({"role": "assistant", "content": ai_reply})
        # Cap history to prevent unbounded memory growth
        if len(hist) > _MAX_HISTORY_TURNS * 2:
            _call_history[call_sid] = hist[-(_MAX_HISTORY_TURNS * 2):]

    log.info("👤 USER: %s", user_input)
    log.info("🤖 AGENT: %s", ai_reply)
    log.info("⏱  pipeline: %.3fs", time.perf_counter() - t0)
    return filenames, ai_reply


# ── Core voice handler ─────────────────────────────────────────────────────────

def _is_inbound(direction: str | None) -> bool:
    return not direction or direction.strip().lower() == "inbound"


def _form_get(form_like, key: str, default=None):
    v = form_like.get(key)
    if v is not None and str(v).strip():
        return v
    v = form_like.get(key.lower())
    if v is not None and str(v).strip():
        return v
    return default


async def _handle_voice(form_like, tenant_id: str | None = None) -> Response:
    t0        = time.perf_counter()
    ngrok_url = os.getenv("NGROK_URL", "").rstrip("/")
    call_sid  = (_form_get(form_like, "CallSid") or "").strip()
    direction = _form_get(form_like, "Direction") or ""
    speech    = _form_get(form_like, "SpeechResult")
    conf      = _form_get(form_like, "Confidence")

    log.info("▶ turn: sid=%s tenant=%s dir=%s conf=%s speech=%s",
             call_sid, tenant_id, direction, conf, (speech or "")[:80])

    if call_sid and tenant_id:
        _call_tenant_map[call_sid] = tenant_id

    try:
        history = _call_history.get(call_sid, [])
        is_cont = len(history) > 0

        if not speech or not speech.strip():
            user_input = "[unclear]" if is_cont else (
                "[inbound_start]" if _is_inbound(direction) else "[outbound_start]"
            )
        else:
            user_input = speech.strip()

        audio_filenames: list[str] = []
        ai_reply = None

        # Fast path: unclear speech
        if user_input == "[unclear]":
            for key in ("unclear_hi", "unclear_en"):
                fn = _cache_consume(FIXED_RESPONSES[key])
                if fn:
                    audio_filenames = [fn]
                    ai_reply        = FIXED_RESPONSES[key]
                    break

        if not audio_filenames:
            base_prompt = await _get_base_prompt(tenant_id)

            # ── RAG query strategy ─────────────────────────────────────────
            # On call start: broad query + high top_k to load full script
            # On user speech: use exact speech as query for precise retrieval
            # ──────────────────────────────────────────────────────────────
            if user_input in ("[outbound_start]", "[inbound_start]"):
                rag_query = "company services questions details process customer information"
                top_k     = 12
            elif user_input == "[unclear]":
                rag_query = "clarify repeat"
                top_k     = 2
            else:
                rag_query = user_input
                top_k     = 8

            context = ""
            if tenant_id:
                context = await retrieve_context(
                    tenant_id, rag_query, top_k=top_k, max_chars=4000
                )
                if context:
                    log.info("✅ RAG: tenant=%s chunks retrieved for query='%s'",
                             tenant_id, rag_query[:50])
                else:
                    log.warning(
                        "⚠️  RAG: NO context retrieved for tenant=%s query='%s' — "
                        "agent will have no knowledge. Check document was uploaded.",
                        tenant_id, rag_query[:50],
                    )
            else:
                log.warning("⚠️  No tenant_id — agent has no knowledge base!")

            # build_rag_system_prompt puts knowledge base FIRST in the prompt
            system_prompt = build_rag_system_prompt(base_prompt, context)

            if user_input == "[inbound_start]":
                system_prompt += (
                    "\n\n[NOTE: INBOUND call — user called in. "
                    "Use INBOUND opening. Ask language preference ONCE in Hindi.]"
                )
            elif user_input == "[outbound_start]":
                system_prompt += (
                    "\n\n[NOTE: OUTBOUND call — agent called user. "
                    "Use OUTBOUND opening. Ask language preference ONCE in Hindi.]"
                )
            elif history:
                system_prompt += (
                    "\n\n[NOTE: Call is already in progress. "
                    "Language preference was already asked. DO NOT ask again. "
                    "Continue with the Knowledge Base questions.]"
                )

            messages = [{"role": "system", "content": system_prompt}]
            for msg in history[-6:]:
                messages.append(msg)
            messages.append({"role": "user", "content": user_input})

            audio_filenames, ai_reply = await _run_pipeline(messages, call_sid, user_input)

        action_url = f"{ngrok_url}/voice"
        if tenant_id:
            action_url += f"?tenant_id={tenant_id}"

        play_elements = "".join(
            f"<Play>{ngrok_url}/audio/{fn}</Play>" for fn in audio_filenames
        )
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    {play_elements}
    <Gather input="speech" action="{action_url}" method="POST"
            speechTimeout="0.8" timeout="5" actionOnEmptyResult="true"
            language="hi-IN" enhanced="true" />
</Response>"""
        log.info("⏱  handler total: %.3fs", time.perf_counter() - t0)
        return Response(content=twiml, media_type="text/xml")

    except Exception as e:
        log.exception("Voice handler error: %s", e)
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="hi-IN">Technical issue. Please try again.</Say>
    <Gather input="speech" action="{ngrok_url}/voice" method="POST"
            speechTimeout="0.8" timeout="5" actionOnEmptyResult="true" language="hi-IN" />
</Response>"""
        return Response(content=twiml, media_type="text/xml")


# ── Startup / Shutdown ─────────────────────────────────────────────────────────

def _cleanup_stale_audio(max_age_seconds: int = 300):
    if not os.path.isdir(AUDIO_DIR):
        return
    now, removed = time.time(), 0
    for name in os.listdir(AUDIO_DIR):
        path = os.path.join(AUDIO_DIR, name)
        try:
            if os.path.isfile(path) and (now - os.path.getmtime(path)) > max_age_seconds:
                os.remove(path)
                removed += 1
        except OSError as err:
            log.warning("Cleanup: %s — %s", name, err)
    if removed:
        log.info("Cleaned %d stale audio file(s)", removed)


async def _cleanup_audio_loop():
    while True:
        await asyncio.sleep(120)
        _cleanup_stale_audio()


@app.on_event("startup")
async def startup():
    log.info("Starting up — RAG + TTS warm-up…")
    rag_task = asyncio.create_task(rag_startup())
    await warmup_ws()
    await rag_task
    asyncio.create_task(_prerender_all())
    asyncio.create_task(_cleanup_audio_loop())
    log.info("✅ Startup complete.")


@app.on_event("shutdown")
async def shutdown():
    try:
        from stt import close_stt_client
        await close_stt_client()
    except Exception:
        pass
    await close_http_client()
    log.info("Shutdown complete.")


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "cached_responses": len(_audio_cache)}


# ── Voice routes ───────────────────────────────────────────────────────────────

@app.post("/voice")
async def voice_webhook(request: Request):
    """Twilio webhook. Set to: https://your-ngrok/voice?tenant_id=sharma_logistics"""
    form      = await request.form()
    params    = dict(request.query_params)
    tenant_id = params.get("tenant_id") or request.headers.get("X-Tenant-ID")
    return await _handle_voice({**params, **dict(form)}, tenant_id=tenant_id)


@app.post("/voice-stream")
async def voice_stream_webhook(request: Request):
    ngrok_url = os.getenv("NGROK_URL", "").rstrip("/")
    stream_ws = ngrok_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    if not stream_ws.endswith("/media-stream"):
        stream_ws = stream_ws.rstrip("/") + "/media-stream"
    tenant_id = dict(request.query_params).get("tenant_id") or request.headers.get("X-Tenant-ID")
    param_tag = ""
    if tenant_id:
        param_tag = f'<Parameter name="tenant_id" value="{tenant_id}"/>'
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect><Stream url="{stream_ws}">{param_tag}</Stream></Connect>
</Response>"""
    return Response(content=twiml, media_type="text/xml")


@app.websocket("/media-stream")
async def media_stream_ws(websocket: WebSocket):
    tenant_id     = websocket.query_params.get("tenant_id")
    base_prompt   = await _get_base_prompt(tenant_id)
    # For media streams, retrieve broad context upfront
    context = ""
    if tenant_id:
        context = await retrieve_context(
            tenant_id,
            "company services questions details process customer information",
            top_k=12, max_chars=4000,
        )
    system_prompt = build_rag_system_prompt(base_prompt, context)
    await handle_media_stream(
        websocket, system_prompt, _call_history, groq_client,
        tenant_id=tenant_id,
        call_tenant_map=_call_tenant_map,
    )


@app.post("/voice/fallback")
async def voice_fallback():
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="hi-IN">Could not connect. Goodbye.</Say>
    <Hangup/>
</Response>"""
    return Response(content=twiml, media_type="text/xml")


@app.post("/voice/status")
async def voice_status(request: Request):
    try:
        body     = dict(await request.form())
        call_sid = body.get("CallSid")
        status   = (body.get("CallStatus") or "").strip().lower()
        log.info("Call status: %s → %s", call_sid, status)
        # FIX: clean up on ALL terminal statuses, not just "completed"
        if call_sid and status in _TERMINAL_CALL_STATUSES:
            _call_history.pop(call_sid, None)
            _call_tenant_map.pop(call_sid, None)
    except Exception:
        pass
    return Response(content="", status_code=200)


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    if not (filename.endswith(".mp3") or filename.endswith(".wav")):
        return Response(status_code=404)
    if ".." in filename or "/" in filename:
        return Response(status_code=404)
    filepath = os.path.join(AUDIO_DIR, filename)
    if not os.path.isfile(filepath):
        return Response(status_code=404)
    with open(filepath, "rb") as f:
        data = f.read()
    # FIX: do NOT delete immediately — Twilio may retry the URL on network hiccups.
    # File will be cleaned up by _cleanup_stale_audio() after 5 minutes.
    mime = "audio/basic" if filename.endswith(".wav") else "audio/mpeg"
    return Response(content=data, media_type=mime)


# ── Outbound call ──────────────────────────────────────────────────────────────

@app.post("/call-user")
async def call_user(mobile_number: str, tenant_id: str):
    """POST /call-user?mobile_number=+918319644992&tenant_id=sharma_logistic"""
    account_sid   = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token    = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
    ngrok_url     = os.getenv("NGROK_URL", "").rstrip("/")

    if not ngrok_url:
        return JSONResponse({"error": "NGROK_URL not set"}, status_code=400)
    if not all([account_sid, auth_token, twilio_number]):
        return JSONResponse({"error": "Twilio credentials not configured"}, status_code=400)

    agent_name   = ""
    tenant = await get_tenant(tenant_id)
    if tenant:
        agent_name = (tenant.get("agent_name") or "").strip()

    if not agent_name:
        return JSONResponse(
            {"error": "Tenant has no agent_name configured"}, status_code=400
        )

    company_name = ""
    context = await retrieve_context(tenant_id, "company name overview", top_k=3, max_chars=1000)
    if context:
        for line in context.split("\n"):
            stripped = line.strip()
            if stripped and len(stripped) > 3 and not stripped.startswith("•"):
                company_name = stripped.split("–")[0].split("—")[0].split("-")[0].strip()
                if 3 < len(company_name) < 60:
                    break
                company_name = ""

    if company_name:
        opening_text = (
            f"Hello! मैं {agent_name} बोल रही हूं, {company_name} की तरफ से। "
            f"आप हिंदी में बात करना पसंद करेंगे या English में?"
        )
    else:
        opening_text = (
            f"Hello! मैं {agent_name} बोल रही हूं। "
            f"आप हिंदी में बात करना पसंद करेंगे या English में?"
        )

    log.info("🤖 AGENT (opening): %s", opening_text)

    try:
        audio_filename = await synthesise_text(opening_text)
    except Exception as e:
        log.exception("TTS failed for opening text: %r — error: %s", opening_text, e)
        return JSONResponse({"error": f"TTS failed: {e}"}, status_code=500)

    audio_url = f"{ngrok_url}/audio/{audio_filename}"
    stream_ws = ngrok_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    if not stream_ws.endswith("/media-stream"):
        stream_ws = stream_ws.rstrip("/") + "/media-stream"

    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"<Play>{audio_url}</Play>"
        "<Connect>"
        f'<Stream url="{stream_ws}">'
        f'<Parameter name="tenant_id" value="{tenant_id}"/>'
        "</Stream>"
        "</Connect>"
        "</Response>"
    )

    twilio_client = Client(account_sid, auth_token)
    call = twilio_client.calls.create(
        to=mobile_number, from_=twilio_number, twiml=twiml,
        status_callback=f"{ngrok_url}/voice/status",
        status_callback_event=["completed", "busy", "no-answer", "failed"],
    )
    _call_history[call.sid] = [
        {"role": "assistant", "content": opening_text},
    ]
    _call_tenant_map[call.sid] = tenant_id

    log.info("Outbound call: %s → %s (tenant=%s)", call.sid, mobile_number, tenant_id)
    return {"status": "calling", "call_sid": call.sid, "tenant_id": tenant_id}


# ── Tenant document routes ─────────────────────────────────────────────────────

@app.post("/tenant/{tenant_id}/documents")
async def upload_document(
    tenant_id: str,
    file: UploadFile = File(...),
    x_api_key: str = Header(None, alias="X-API-Key"),
):
    await _require_tenant_auth(tenant_id, x_api_key)
    file_bytes   = await file.read()
    content_type = file.content_type or ""
    filename     = file.filename or "document"
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")
    result = await ingest_file_bytes(tenant_id, file_bytes, filename, content_type)
    log.info("Document uploaded: tenant=%s file=%s chunks=%d",
             tenant_id, filename, result["chunks_added"])
    return result


@app.post("/tenant/{tenant_id}/documents/text")
async def upload_text(
    tenant_id: str,
    body: UploadTextRequest,
    x_api_key: str = Header(None, alias="X-API-Key"),
):
    await _require_tenant_auth(tenant_id, x_api_key)
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="'text' field is empty")
    return await ingest_text(tenant_id, text, source_name=body.source_name)


@app.get("/tenant/{tenant_id}/documents")
async def list_documents(
    tenant_id: str,
    x_api_key: str = Header(None, alias="X-API-Key"),
):
    await _require_tenant_auth(tenant_id, x_api_key)
    docs = await list_tenant_documents(tenant_id)
    return {"tenant_id": tenant_id, "documents": docs}


@app.delete("/tenant/{tenant_id}/documents/{doc_id}")
async def delete_document(
    tenant_id: str,
    doc_id: str,
    x_api_key: str = Header(None, alias="X-API-Key"),
):
    await _require_tenant_auth(tenant_id, x_api_key)
    count = await delete_tenant_documents(tenant_id, doc_id)
    return {"deleted_chunks": count}


@app.delete("/tenant/{tenant_id}/documents")
async def delete_all_documents(
    tenant_id: str,
    x_api_key: str = Header(None, alias="X-API-Key"),
):
    await _require_tenant_auth(tenant_id, x_api_key)
    await delete_tenant_collection(tenant_id)
    return {"status": "deleted", "tenant_id": tenant_id}


# ── Admin routes ───────────────────────────────────────────────────────────────

@app.post("/admin/tenants")
async def admin_create_tenant(
    body: CreateTenantRequest,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    _require_admin(x_admin_key)
    try:
        return await create_tenant(
            tenant_id              = body.tenant_id,
            name                   = body.name,
            agent_name             = body.agent_name,
            language_preference    = body.language_preference,
            system_prompt_override = body.system_prompt_override,
            extra_context          = body.extra_context,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/admin/tenants")
async def admin_list_tenants(x_admin_key: str = Header(None, alias="X-Admin-Key")):
    _require_admin(x_admin_key)
    return {"tenants": await list_tenants()}


@app.get("/admin/tenants/{tenant_id}")
async def admin_get_tenant(
    tenant_id: str,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    _require_admin(x_admin_key)
    t = await get_tenant(tenant_id)
    if not t:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return {k: v for k, v in t.items() if k != "api_key_hash"}


@app.patch("/admin/tenants/{tenant_id}")
async def admin_update_tenant(
    tenant_id: str,
    body: UpdateTenantRequest,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    _require_admin(x_admin_key)
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    result  = await update_tenant(tenant_id, **updates)
    if not result:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return result


@app.delete("/admin/tenants/{tenant_id}")
async def admin_delete_tenant(
    tenant_id: str,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    _require_admin(x_admin_key)
    await delete_tenant_collection(tenant_id)
    deleted = await delete_tenant(tenant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return {"status": "deleted", "tenant_id": tenant_id}


@app.post("/admin/tenants/{tenant_id}/rotate-key")
async def admin_rotate_key(
    tenant_id: str,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    _require_admin(x_admin_key)
    new_key = await rotate_api_key(tenant_id)
    if not new_key:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return {"tenant_id": tenant_id, "api_key": new_key}