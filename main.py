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
  4. On each turn: STT → RAG retrieve → LLM (grounded) → TTS → Twilio
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

_call_history:   dict[str, list[dict]] = {}   # call_sid → message history
_call_tenant_map: dict[str, str]       = {}   # call_sid → tenant_id
_audio_cache:    dict[str, str]        = {}   # text → mp3 filename

_SENTENCE_END     = re.compile(r"[।.?!]\s*")
_MAX_BUFFER_CHARS = 120

# ── Pydantic request models ────────────────────────────────────────────────────

class CreateTenantRequest(BaseModel):
    tenant_id:              str           = Field(...,    description="Unique ID e.g. sharma_logistics")
    name:                   str           = Field(...,    description="Display name e.g. Sharma Logistics")
    agent_name:             str           = Field("SONY", description="AI agent name spoken on calls")
    language_preference:    str           = Field("hi-IN",description="Default language code e.g. hi-IN or en-IN")
    system_prompt_override: Optional[str] = Field(None,   description="Custom system prompt — leave blank to use default")
    extra_context:          str           = Field("",     description="Static text always injected e.g. disclaimers, service area")


class UpdateTenantRequest(BaseModel):
    name:                   Optional[str]  = Field(None, description="Display name")
    agent_name:             Optional[str]  = Field(None, description="AI agent name")
    language_preference:    Optional[str]  = Field(None, description="Language code")
    system_prompt_override: Optional[str]  = Field(None, description="Full custom system prompt override")
    extra_context:          Optional[str]  = Field(None, description="Static context always injected")
    active:                 Optional[bool] = Field(None, description="Enable or disable this tenant")


class UploadTextRequest(BaseModel):
    text:        str = Field(...,             description="Raw text to ingest into the knowledge base")
    source_name: str = Field("text_upload",   description="Label for this document e.g. pricing_guide")


# ── System prompt ──────────────────────────────────────────────────────────────
# Behaviour-only. ALL factual knowledge comes from the RAG KNOWLEDGE BASE injected
# at runtime from the tenant's uploaded documents. Agent never uses training data.

BASE_SYSTEM_PROMPT = """You are {agent_name}, an AI voice agent working on behalf of {company_name}.

You have two jobs on this call:
1. If the user has a problem or complaint — listen carefully, understand the issue fully, and guide them to a resolution using the Knowledge Base.
2. If the user has no problem — introduce the services of {company_name} and guide them step by step using the Knowledge Base.

Every answer you give must come ONLY from the KNOWLEDGE BASE provided at the bottom of this prompt.

---

LANGUAGE — follow strictly:
The very first thing you do on every call is ask the user which language they prefer.
Say exactly: "नमस्ते! आप किस भाषा में बात करना चाहेंगे — हिंदी या इंग्लिश? / Hello! Which language would you prefer — Hindi or English?"
Wait for the user to answer. Once they choose, speak only in that language for the rest of the call.
Never switch languages unless the user does first.
If the user's choice is unclear, ask again: "क्या आप हिंदी में बात करना चाहेंगे या English में?"

---

VOICE CALL STYLE — follow strictly:
This is a live phone call. Keep every reply to 2 or 3 short spoken sentences maximum.
Never use bullet points, numbered lists, or any text formatting.
Speak naturally, warmly, and clearly like a helpful human agent.
Ask only one question at a time and wait for the answer before moving on.
Never repeat what the user has already confirmed.

---

KNOWLEDGE BASE RULES — most important:
The KNOWLEDGE BASE section below is your only source of truth.
If the answer is there → answer directly and accurately from it.
If the answer is NOT there → never guess or invent. Say:
  Hindi:   "मुझे इस बारे में अभी सही जानकारी नहीं है, मैं टीम से कन्फर्म करके आपको बताऊंगा।"
  English: "I don't have the exact details on that right now, I'll confirm with the team and get back to you."
Never answer from your own training knowledge. Never invent facts, prices, steps, policies, or names.

---

CALL FLOW — follow in order, never skip:

STEP 1 — Ask language preference (always first, before anything else).

STEP 2 — After language confirmed, greet in chosen language and ask:
  Hindi:   "क्या आपको हमारी किसी सर्विस से कोई समस्या है, या आप हमारी सर्विस के बारे में जानकारी लेना चाहते हैं?"
  English: "Are you calling about an issue with our service, or would you like to know more about what we offer?"

STEP 3 — Handle the call using Knowledge Base only:
  Problem → understand it with one question at a time, then resolve using Knowledge Base.
  Service enquiry → explain relevant info from Knowledge Base and guide next steps.

STEP 4 — Close warmly:
  Hindi:   "धन्यवाद आपके समय के लिए। जल्द ही हमारी टीम आपसे संपर्क करेगी। आपका दिन शुभ हो।"
  English: "Thank you for your time. Our team will get back to you soon. Have a great day."

If at any point the user says it is not a good time:
  Say "कोई बात नहीं, जब भी सुविधा हो तब कॉल करें, धन्यवाद।" and end the call politely.
"""

# ── Fixed responses & opening audio ───────────────────────────────────────────

FIXED_RESPONSES: dict[str, str] = {
    "lang_ask":     "नमस्ते! आप किस भाषा में बात करना चाहेंगे — हिंदी या इंग्लिश? Hello! Which language would you prefer — Hindi or English?",
    "unclear_hi":   "कृपया दोबारा बोलें।",
    "unclear_en":   "Could you please repeat that?",
    "unclear_both": "Sorry, I didn't catch that. क्या आप दोबारा बोल सकते हैं?",
}

OUTBOUND_OPENING = (
    "नमस्ते! आप किस भाषा में बात करना चाहेंगे — हिंदी या इंग्लिश? "
    "Hello! Which language would you prefer — Hindi or English?"
)

# ── Auth helpers ───────────────────────────────────────────────────────────────

def _require_admin(api_key: str | None):
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY not configured on server")
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

async def _get_tenant_system_prompt(tenant_id: str | None) -> str:
    """Build system prompt for tenant — substitutes agent_name and company_name."""
    defaults = {"agent_name": "SONY", "company_name": "our company"}

    if not tenant_id:
        return BASE_SYSTEM_PROMPT.replace("{agent_name}", defaults["agent_name"]) \
                                 .replace("{company_name}", defaults["company_name"])

    tenant = await get_tenant(tenant_id)
    if not tenant:
        return BASE_SYSTEM_PROMPT.replace("{agent_name}", defaults["agent_name"]) \
                                 .replace("{company_name}", defaults["company_name"])

    agent_name   = tenant.get("agent_name", defaults["agent_name"])
    company_name = tenant.get("name",       defaults["company_name"])

    if tenant.get("system_prompt_override"):
        prompt = tenant["system_prompt_override"]
    else:
        prompt = BASE_SYSTEM_PROMPT.replace("{agent_name}", agent_name) \
                                   .replace("{company_name}", company_name)

    extra = tenant.get("extra_context", "").strip()
    if extra:
        prompt += f"\n\n--- ADDITIONAL CONTEXT ---\n{extra}\n---"

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
        asyncio.create_task(_rerender_bg(text))
    return fn


async def _rerender_bg(text: str):
    try:
        fn = await synthesise_text(text)
        _audio_cache[text.strip()] = fn
    except Exception as e:
        log.warning("Cache re-render failed: %s", e)


async def _prerender_all():
    for key, text in FIXED_RESPONSES.items():
        try:
            fn = await synthesise_text(text)
            _audio_cache[text.strip()] = fn
            log.info("  ✅ [%s] cached", key)
        except Exception as e:
            log.warning("  ❌ [%s] failed: %s", key, e)

# ── LLM streaming helpers ──────────────────────────────────────────────────────

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
    sentences: list[str] = []

    async for sentence in _stream_llm_sentences(messages):
        if sentence.strip():
            sentences.append(sentence.strip())

    ai_reply = " ".join(sentences).strip() or "कृपया दोबारा बोलें।"

    if not sentences:
        stream = await groq_client.chat.completions.create(
            model=GROQ_MODEL, messages=messages,
            max_tokens=128, temperature=0.1, stream=True, tool_choice="none",
        )
        fn, ai_reply = await llm_to_tts_stream(stream)
        ai_reply     = ai_reply.strip() or "कृपया दोबारा बोलें।"
        filenames    = [fn]
    else:
        tts_tasks = [generate_tts_async(s) for s in sentences]
        filenames = list(await asyncio.gather(*tts_tasks))
        log.info("⏱  parallel TTS: %.3fs | %d segments", time.perf_counter() - t0, len(filenames))

    if call_sid:
        hist = _call_history.setdefault(call_sid, [])
        hist.append({"role": "user",      "content": user_input})
        hist.append({"role": "assistant", "content": ai_reply})

    log.info("⏱  pipeline: %.3fs | %s", time.perf_counter() - t0, ai_reply[:80])
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

        # Fast path for unclear speech
        audio_filenames: list[str] = []
        ai_reply = None

        if user_input == "[unclear]":
            for key in ("unclear_hi", "unclear_en"):
                fn = _cache_consume(FIXED_RESPONSES[key])
                if fn:
                    audio_filenames = [fn]
                    ai_reply        = FIXED_RESPONSES[key]
                    break

        if not audio_filenames:
            base_prompt = await _get_tenant_system_prompt(tenant_id)

            # RAG retrieval — always on, agent relies 100% on documents
            rag_query = user_input
            if user_input in ("[inbound_start]", "[outbound_start]"):
                rag_query = "greeting introduction company overview services"
            elif user_input == "[unclear]":
                rag_query = "please repeat clarify"

            context = ""
            if tenant_id:
                context = await retrieve_context(tenant_id, rag_query)
                if context:
                    log.info("RAG: injected %d chars for query='%s'", len(context), rag_query[:50])
                else:
                    log.info("RAG: no matching context for query='%s'", rag_query[:50])

            system_prompt = build_rag_system_prompt(base_prompt, context)

            messages = [{"role": "system", "content": system_prompt}]
            for msg in history[-6:]:
                messages.append(msg)
            messages.append({"role": "user", "content": user_input})

            audio_filenames, ai_reply = await _run_pipeline(messages, call_sid, user_input)

        # Build action URL with tenant_id so it survives Twilio's redirect
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
        if not name.endswith(".mp3"):
            continue
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
    log.info("✅ Startup complete — RAG + voice pipeline ready.")


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

# ── Voice routes (Twilio webhooks) ─────────────────────────────────────────────

@app.post("/voice")
async def voice_webhook(request: Request):
    """
    Twilio Gather webhook for inbound calls.
    Set your Twilio phone number webhook to:
      https://your-ngrok-url/voice?tenant_id=sharma_logistics
    """
    form      = await request.form()
    params    = dict(request.query_params)
    tenant_id = params.get("tenant_id") or request.headers.get("X-Tenant-ID")
    return await _handle_voice({**params, **dict(form)}, tenant_id=tenant_id)


@app.post("/voice-stream")
async def voice_stream_webhook(request: Request):
    """
    Twilio webhook — returns TwiML <Stream> for Media Streams (real-time audio).
    Set your Twilio webhook to:
      https://your-ngrok-url/voice-stream?tenant_id=sharma_logistics
    """
    ngrok_url = os.getenv("NGROK_URL", "").rstrip("/")
    stream_ws = ngrok_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    if not stream_ws.endswith("/media-stream"):
        stream_ws = stream_ws.rstrip("/") + "/media-stream"

    tenant_id = dict(request.query_params).get("tenant_id") or request.headers.get("X-Tenant-ID")
    if tenant_id:
        stream_ws += f"?tenant_id={tenant_id}"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{stream_ws}" />
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="text/xml")


@app.websocket("/media-stream")
async def media_stream_ws(websocket: WebSocket):
    tenant_id     = websocket.query_params.get("tenant_id")
    system_prompt = await _get_tenant_system_prompt(tenant_id)
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
        if call_sid and status == "completed":
            _call_history.pop(call_sid, None)
            _call_tenant_map.pop(call_sid, None)
    except Exception:
        pass
    return Response(content="", status_code=200)


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    if not filename.endswith(".mp3") or ".." in filename or "/" in filename:
        return Response(status_code=404)
    filepath = os.path.join(AUDIO_DIR, filename)
    if not os.path.isfile(filepath):
        return Response(status_code=404)
    with open(filepath, "rb") as f:
        data = f.read()
    try:
        os.remove(filepath)
    except OSError:
        pass
    return Response(content=data, media_type="audio/mpeg")

# ── Outbound call ──────────────────────────────────────────────────────────────

@app.post("/call-user")
async def call_user(
    mobile_number: str,
    tenant_id: str,
):
    """
    Initiate an outbound call to a user.

    Query parameters (both required):
      - mobile_number: e.g. +918319644992
      - tenant_id:     e.g. sharma_logistics

    Example:
      POST /call-user?mobile_number=+918319644992&tenant_id=sharma_logistics
    """
    account_sid   = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token    = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
    ngrok_url     = os.getenv("NGROK_URL", "").rstrip("/")

    if not ngrok_url:
        return JSONResponse({"error": "NGROK_URL not set in .env"}, status_code=400)
    if not all([account_sid, auth_token, twilio_number]):
        return JSONResponse({"error": "Twilio credentials not configured"}, status_code=400)

    # Synthesise opening audio (TTS cache makes this fast after first call)
    try:
        audio_filename = await synthesise_text(OUTBOUND_OPENING)
    except Exception as e:
        return JSONResponse({"error": f"TTS failed: {e}"}, status_code=500)

    audio_url = f"{ngrok_url}/audio/{audio_filename}"
    stream_ws = ngrok_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    if not stream_ws.endswith("/media-stream"):
        stream_ws = stream_ws.rstrip("/") + "/media-stream"
    stream_ws += f"?tenant_id={tenant_id}"

    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"<Play>{audio_url}</Play>"
        f'<Connect><Stream url="{stream_ws}" /></Connect>'
        "</Response>"
    )

    twilio_client = Client(account_sid, auth_token)
    call = twilio_client.calls.create(
        to=mobile_number,
        from_=twilio_number,
        twiml=twiml,
        status_callback=f"{ngrok_url}/voice/status",
        status_callback_event=["completed"],
    )
    log.info("Outbound call initiated: %s → %s (tenant=%s)", call.sid, mobile_number, tenant_id)
    return {"status": "calling", "call_sid": call.sid, "tenant_id": tenant_id}

# ── Tenant document upload ─────────────────────────────────────────────────────

@app.post("/tenant/{tenant_id}/documents")
async def upload_document(
    tenant_id: str,
    file: UploadFile = File(...),
    x_api_key: str = Header(None, alias="X-API-Key"),
):
    """
    Upload a document (PDF, TXT, MD, CSV) to a tenant's knowledge base.
    Requires X-API-Key header matching the tenant's API key.
    """
    await _require_tenant_auth(tenant_id, x_api_key)

    file_bytes   = await file.read()
    content_type = file.content_type or ""
    filename     = file.filename or "document"

    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")

    result = await ingest_file_bytes(tenant_id, file_bytes, filename, content_type)
    log.info("Document uploaded: tenant=%s file=%s chunks=%d", tenant_id, filename, result["chunks_added"])
    return result


@app.post("/tenant/{tenant_id}/documents/text")
async def upload_text(
    tenant_id: str,
    body: UploadTextRequest,
    x_api_key: str = Header(None, alias="X-API-Key"),
):
    """Upload raw text as a document to the knowledge base."""
    await _require_tenant_auth(tenant_id, x_api_key)
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="'text' field is empty")
    result = await ingest_text(tenant_id, text, source_name=body.source_name)
    return result


@app.get("/tenant/{tenant_id}/documents")
async def list_documents(
    tenant_id: str,
    x_api_key: str = Header(None, alias="X-API-Key"),
):
    """List all documents ingested for a tenant."""
    await _require_tenant_auth(tenant_id, x_api_key)
    docs = await list_tenant_documents(tenant_id)
    return {"tenant_id": tenant_id, "documents": docs}


@app.delete("/tenant/{tenant_id}/documents/{doc_id}")
async def delete_document(
    tenant_id: str,
    doc_id: str,
    x_api_key: str = Header(None, alias="X-API-Key"),
):
    """Delete a specific document from the knowledge base."""
    await _require_tenant_auth(tenant_id, x_api_key)
    count = await delete_tenant_documents(tenant_id, doc_id)
    return {"deleted_chunks": count}


@app.delete("/tenant/{tenant_id}/documents")
async def delete_all_documents(
    tenant_id: str,
    x_api_key: str = Header(None, alias="X-API-Key"),
):
    """Wipe all documents for a tenant."""
    await _require_tenant_auth(tenant_id, x_api_key)
    await delete_tenant_collection(tenant_id)
    return {"status": "deleted", "tenant_id": tenant_id}

# ── Admin API ──────────────────────────────────────────────────────────────────

@app.post("/admin/tenants")
async def admin_create_tenant(
    body: CreateTenantRequest,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    """
    Create a new tenant. Requires X-Admin-Key header.
    Returns api_key — save it, shown only once.
    """
    _require_admin(x_admin_key)
    try:
        result = await create_tenant(
            tenant_id              = body.tenant_id,
            name                   = body.name,
            agent_name             = body.agent_name,
            language_preference    = body.language_preference,
            system_prompt_override = body.system_prompt_override,
            extra_context          = body.extra_context,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/admin/tenants")
async def admin_list_tenants(
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    """List all tenants."""
    _require_admin(x_admin_key)
    return {"tenants": await list_tenants()}


@app.get("/admin/tenants/{tenant_id}")
async def admin_get_tenant(
    tenant_id: str,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    """Get a specific tenant's details."""
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
    """Update tenant fields. Only provided fields are updated."""
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
    """Delete a tenant and all their documents."""
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
    """Rotate a tenant's API key. Old key immediately invalidated."""
    _require_admin(x_admin_key)
    new_key = await rotate_api_key(tenant_id)
    if not new_key:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return {"tenant_id": tenant_id, "api_key": new_key}