import logging
import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
from groq import Groq
from twilio.rest import Client
from tts import generate_tts, AUDIO_DIR

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

_call_history: dict[str, list[dict[str, str]]] = {}

SYSTEM_PROMPT = """

CRITICAL LANGUAGE POLICY (MANDATORY – OVERRIDES ALL OTHER INSTRUCTIONS):

1. You MUST always reply in the same language as the user's most recent message.
2. If the user speaks Hindi, respond completely in Hindi.
3. If the user speaks English, respond completely in English.
4. If the user speaks in mixed Hindi and English (Hinglish), respond in simple Hinglish.
5. Never default to English automatically.
6. If you are unsure about the user's preferred language, ask:
   "आप किस भाषा में बात करना चाहेंगे — हिंदी या इंग्लिश?"
7. Once the user selects a language, continue the entire conversation strictly in that language unless the user switches.
8. Keep all responses short (maximum 2–3 sentences) and suitable for a phone conversation.

------------------------------------------------------------

You are SONY, a polite and professional AI voice assistant representing Sharma Logistics.

Your goal is to:
- Qualify a household shifting enquiry
- Collect required details
- Build trust
- Schedule a free home survey

Always speak clearly, naturally, patiently, and respectfully.
Maintain a warm, helpful, and professional tone at all times.
Keep responses concise and conversational (no long paragraphs).

------------------------------------------------------------
CONVERSATION FLOW

OPENING (For inbound calls):
नमस्ते। मैं सोनी बोल रही हूं, शर्मा लॉजिस्टिक्स की तरफ से। मैं आपके घर शिफ्टिंग की इन्क्वायरी में मदद करने वाली AI सहायक हूं। क्या अभी आप एक मिनट बात कर सकते हैं?

OPENING (For outbound calls):
नमस्ते। मैं सोनी बोल रही हूं, शर्मा लॉजिस्टिक्स की तरफ से। मैं आपके घर शिफ्टिंग की इन्क्वायरी के संबंध में कॉल कर रही हूं। क्या अभी आप एक मिनट बात कर सकते हैं?

If the user says it is NOT a good time:
कोई बात नहीं। जब भी सुविधा हो कृपया कॉल कर लें। धन्यवाद।
(Politely end the call.)

If user agrees to talk:
आगे बढ़ने से पहले, आप किस भाषा में बात करना पसंद करेंगे — हिंदी या इंग्लिश?

------------------------------------------------------------
PURPOSE OF THE CALL

If Hindi selected:
धन्यवाद। मैं आपकी इन्दौर, मध्य प्रदेश से पुणे, महाराष्ट्र तक घर का सामान शिफ्ट करने की इन्क्वायरी के बारे में कॉल कर रही हूं। बस कुछ बातें पक्की कर लूं ताकि हम ठीक से मदद कर सकें।

If English selected:
Thank you. I am calling regarding your enquiry for shifting your household items from Indore, Madhya Pradesh to Pune, Maharashtra. I just need to confirm a few details so that we can assist you properly.

------------------------------------------------------------
QUESTIONS FLOW (Ask one at a time, wait for response)

Q1 – Branch Contact Status  
Hindi: क्या हमारी ब्रांच से किसी ने आपको पहले कॉल किया है और कोटेशन भेजा है?  
English: Has anyone from our branch already called you and shared a quotation?

If NO:
Hindi: देरी के लिए माफी चाहती हूं। हम तुरंत आपकी मदद करेंगे।  
English: I sincerely apologize for the delay. We will assist you immediately.

------------------------------------------------------------

Q2 – Household Size  
Hindi: आप एक BHK, दो BHK या तीन BHK शिफ्ट कर रहे हैं?  
English: Are you shifting a 1 BHK, 2 BHK, or 3 BHK household?

------------------------------------------------------------

Q3 – Move Details  
Hindi: पिकअप का फ्लोर नंबर क्या है? लिफ्ट है या नहीं? और क्या कोई गाड़ी शिफ्ट करनी है?  
English: What is the pickup floor number? Is there a lift? Are any vehicles being shifted?

------------------------------------------------------------

Q4 – Quotation Preference  
Hindi: आप कोटेशन ईमेल पर चाहेंगे या व्हाट्सऐप पर?  
English: Would you like the quotation on email or WhatsApp?

------------------------------------------------------------

Q5 – Address Collection  
Hindi: कृपया पूरा पिकअप पता पिनकोड सहित बताएं।  
English: Please share the complete pickup address with pincode.

------------------------------------------------------------

Q6 – Survey Scheduling  
Hindi: किस दिन और समय पर सर्वे के लिए सुविधाजनक रहेगा?  
English: Which day and time would be convenient for the survey?

------------------------------------------------------------

TRUST BUILDING STATEMENT

Hindi:
सर्वे के दौरान हमारा फील्ड ऑफिसर सुरक्षित पैकिंग, इंश्योरेंस विकल्प और पारदर्शी कोटेशन की पूरी जानकारी देगा। कोई छुपा चार्ज नहीं होगा।

English:
During the survey, our field officer will explain safe packing, insurance options, and provide a transparent quotation with no hidden charges.

------------------------------------------------------------

If user is concerned about price:

Hindi:
आप सिर्फ उतने सामान का भुगतान करेंगे जितना आप शिफ्ट करवाते हैं। अंतिम कोटेशन सामान और दूरी के अनुसार होगा।

English:
You only pay for the items you move. The final quotation depends on the items and distance.

------------------------------------------------------------

CLOSING

Hindi:
आपका समय देने के लिए धन्यवाद। मैंने आपकी जानकारी नोट कर ली है और सर्वे शेड्यूल कर दिया है। आपका दिन शुभ रहे।

English:
Thank you for your time. I have noted your details and scheduled the survey. Have a great day.
"""

def _is_inbound(direction: str | None) -> bool:
    """True if the user called us (inbound); False if we called the user (outbound)."""
    if not direction:
        return True  # default to inbound greeting
    return direction.strip().lower() == "inbound"


def _escape_say(text: str) -> str:
    """Escape text for safe use inside Twilio <Say> (XML)."""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _error_twiml(message: str) -> str:
    """TwiML for exception path only: short Say so user hears something. Normal flow uses Sarvam <Play> only."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="hi-IN">{_escape_say(message)}</Say>
    <Gather input="speech" action="{os.getenv('NGROK_URL', '').rstrip('/')}/voice/start" method="POST" speechTimeout="auto" timeout="3" actionOnEmptyResult="true" language="hi-IN" />
</Response>"""


def _form_get(form_like, key: str, default=None):
    """Get form value; try exact key then lowercase (Twilio sends capitalized)."""
    v = form_like.get(key)
    if v is not None and str(v).strip() != "":
        return v
    if key != key.lower():
        v = form_like.get(key.lower())
        if v is not None and str(v).strip() != "":
            return v
    return default


async def _handle_voice(form_like) -> Response:
    """Shared logic for /voice: form_like must have .get(k, default)."""
    ngrok_url = os.getenv("NGROK_URL", "https://your-ngrok-url").rstrip("/")
    direction = _form_get(form_like, "Direction") or ""
    call_sid = (_form_get(form_like, "CallSid") or "").strip()
    user_input = _form_get(form_like, "SpeechResult")
    confidence = _form_get(form_like, "Confidence")

    log.info(
        "Voice webhook: direction=%s CallSid=%s speech_len=%s confidence=%s",
        direction, call_sid, len((user_input or "").strip()), confidence,
    )

    try:
        history = _call_history.get(call_sid, [])
        is_continuation = len(history) > 0
        if user_input and user_input.strip():
            log.info("Twilio STT: %s (confidence=%s)", user_input.strip()[:160], confidence)

        if not user_input or not user_input.strip():
            if is_continuation:
                user_input = "[Context: The user just spoke but we could not hear them clearly (or the speech result was empty). Politely ask them to repeat what they said, in one short sentence.]"
            else:
                if _is_inbound(direction):
                    user_input = "[Context: The user has just called our company number. Greet them and ask how you can help.]"
                else:
                    user_input = "[Context: You have just called the user. Introduce yourself as SONY from Sharma Logistics and say you are calling regarding their household shifting enquiry. Ask if they have a moment to talk.]"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Keep last 10 turns (20 messages) to avoid token limits
        for msg in history[-20:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_input})

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
        )
        ai_reply = (completion.choices[0].message.content or "").strip()
        if not ai_reply:
            ai_reply = "I did not get a response. Please try again."
        log.info("LLM reply length=%s", len(ai_reply))

        # Append to per-call history so next user reply continues the conversation
        if call_sid:
            if call_sid not in _call_history:
                _call_history[call_sid] = []
            _call_history[call_sid].append({"role": "user", "content": user_input})
            _call_history[call_sid].append({"role": "assistant", "content": ai_reply})

        # Sarvam TTS only — no Twilio <Say>. LLM controls content; Sarvam speaks it.
        audio_filename = generate_tts(ai_reply)
        audio_url = f"{ngrok_url}/audio/{audio_filename}"
        record_action = f"{ngrok_url}/voice/start"
        # Twilio speech gather (no recording fetch / external STT needed).
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Play>{audio_url}</Play>
    <Gather input="speech" action="{record_action}" method="POST" speechTimeout="auto" timeout="3" actionOnEmptyResult="true" language="hi-IN" />
</Response>"""
        return Response(content=twiml, media_type="text/xml")
    except Exception as e:
        log.exception("Voice handler error: %s", e)
        fallback = _error_twiml("We are having a technical issue. Please try again in a moment.")
        return Response(content=fallback, media_type="text/xml")


def _redirect_twiml(start_url: str) -> str:
    """Instant response: redirect to /voice/start so Twilio never times out. No Say."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Redirect method="POST">{start_url}</Redirect>
</Response>"""

@app.get("/")
def server_health():
    return {
        "message": "Server is up and running"
    }

@app.post("/voice")
async def voice_webhook(request: Request):
    """
    First webhook when call connects. Returns immediately (no LLM) then redirects to /voice/start.
    This avoids Twilio timeout while we run LLM. Set this URL in Twilio as the "A call comes in" webhook.
    """
    ngrok_url = os.getenv("NGROK_URL", "https://your-ngrok-url").rstrip("/")
    return Response(
        content=_redirect_twiml(f"{ngrok_url}/voice/start"),
        media_type="text/xml",
    )


@app.post("/voice/start")
async def voice_start_post(request: Request):
    """
    Runs after redirect from /voice and after each user speech gather.
    Does LLM → Sarvam TTS → Play + Gather. Twilio POSTs here (redirect and Gather action).
    """
    form = await request.form()
    return await _handle_voice(form)


@app.post("/voice/fallback")
async def voice_fallback():
    """Optional: set as Fallback URL in Twilio so user hears a message if the main voice URL fails."""
    log.warning("Voice fallback hit - main URL may have failed")
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="hi-IN">We could not connect the assistant. Please try again later. Goodbye.</Say>
    <Hangup/>
</Response>"""
    return Response(content=twiml, media_type="text/xml")


@app.post("/voice/status")
async def voice_status(request: Request):
    """Twilio status callback (answered, completed). Log and clear conversation when call ends."""
    try:
        body = await request.form() if request.method == "POST" else request.query_params
        body_dict = dict(body)
        log.info("Call status: %s", body_dict)
        call_sid = body_dict.get("CallSid")
        status = (body_dict.get("CallStatus") or "").strip().lower()
        if call_sid and status == "completed":
            _call_history.pop(call_sid, None)
    except Exception:
        pass
    return Response(content="", status_code=200)


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve TTS MP3 for Twilio Play; delete from audio_files after use."""
    if not filename.endswith(".mp3") or ".." in filename or "/" in filename:
        return Response(status_code=404)
    filepath = os.path.join(AUDIO_DIR, filename)
    if not os.path.isfile(filepath):
        log.warning("Audio file not found: %s", filepath)
        return Response(status_code=404)
    with open(filepath, "rb") as f:
        audio_bytes = f.read()
    try:
        os.remove(filepath)
    except OSError as e:
        log.warning("Could not delete audio after serve: %s", e)
    return Response(content=audio_bytes, media_type="audio/mpeg")


def _get_first_outbound_message() -> str:
    """Get the LLM's first message for outbound call (used in inline TwiML)."""
    user_ctx = "[Context: You have just called the user. Introduce yourself as SONY from Sharma Logistics and say you are calling regarding their household shifting enquiry. Ask if they have a moment to talk. Keep your reply brief and under 300 words.]"
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_ctx},
        ],
    )
    reply = (completion.choices[0].message.content or "").strip()
    return reply[:3000] if len(reply) > 3000 else reply  # TwiML limit 4000 chars


@app.post("/call-user")
def call_user(mobile_number: str):
    """
    Outbound: AI agent calls the user.
    First message: LLM → Sarvam TTS → inline TwiML with <Play> (no Polly).
    When user answers, Twilio plays Sarvam MP3 then gathers speech (Twilio STT).
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
    ngrok_url = os.getenv("NGROK_URL", "").rstrip("/")

    if not ngrok_url:
        return JSONResponse(content={"error": "NGROK_URL not set in .env"}, status_code=400)

    log.info("Generating first message and Sarvam TTS for outbound call...")
    try:
        first_message = _get_first_outbound_message()
        audio_filename = generate_tts(first_message)
    except Exception as e:
        log.exception("LLM or Sarvam TTS failed for first message: %s", e)
        first_message = "Hello, this is SONY from Sharma Logistics. We are calling about your shifting enquiry. Do you have a moment to talk?"
        try:
            audio_filename = generate_tts(first_message)
        except Exception as tts_err:
            log.exception("Fallback TTS also failed: %s", tts_err)
            return JSONResponse(
                content={"error": "TTS failed. Check SARVAM_API_KEY and Sarvam API (403 = invalid key or quota)."},
                status_code=500,
            )

    audio_url = f"{ngrok_url}/audio/{audio_filename}"
    record_action = f"{ngrok_url}/voice/start"
    # Inline TwiML: Sarvam MP3 only, no <Say>.
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?><Response><Play>{audio_url}</Play><Gather input="speech" action="{record_action}" method="POST" speechTimeout="auto" timeout="3" actionOnEmptyResult="true" language="hi-IN" /></Response>"""

    status_callback = f"{ngrok_url}/voice/status" if ngrok_url else None
    twilio_client = Client(account_sid, auth_token)
    call = twilio_client.calls.create(
        to=mobile_number,
        from_=twilio_number,
        twiml=twiml,
        status_callback=status_callback,
        status_callback_event=["completed"],
    )

    log.info("Outbound call created: %s", call.sid)
    return {"status": "calling", "call_sid": call.sid}
