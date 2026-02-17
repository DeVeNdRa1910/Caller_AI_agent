# Voice AI setup and troubleshooting

## 1. Start the app and ngrok

```bash
# Terminal 1: start the app
./serverstart.sh

# Terminal 2: expose with ngrok (so the phone provider can reach your app)
ngrok http 9000 --request-header-add "ngrok-skip-browser-warning: true"
```

Copy the **HTTPS** URL from ngrok (e.g. `https://xxxx.ngrok-free.app`) and set it in `.env`:

```
NGROK_URL=https://xxxx.ngrok-free.app
```

No trailing slash. Restart the app after changing `.env`.

---

## 2. Test that Twilio can reach your server

Before testing the full AI flow, confirm Twilio gets a valid response:

1. In Twilio Console → Phone Numbers → your number → Voice & Fax.
2. Set **A CALL COMES IN** to:  
   `https://YOUR_NGROK_URL/voice/welcome`  
   Method: **GET** or **POST**.
3. Call your Twilio number from your phone.

You should hear: *"Hello, this is a test from Sharma Logistics. If you hear this, your webhook is working."*

If you hear that, Twilio can reach your app. Set the URL back to `https://YOUR_NGROK_URL/voice` for normal use.

---

## 3. Outbound call (AI calls the user)

Trigger a call from your API, e.g.:

```bash
curl -X POST "http://localhost:9000/call-user?mobile_number=+91XXXXXXXXXX"
```

Or from another app: `POST /call-user?mobile_number=+91XXXXXXXXXX`

When the user answers:

1. Twilio requests `NGROK_URL/voice` → gets an instant reply: "One moment please" + redirect to `/voice/start`.
2. Twilio requests `NGROK_URL/voice/start` → your app runs the LLM and returns TwiML with `<Say>` (SONY’s reply) and `<Record>`.
3. The user hears the LLM reply, then a beep and can speak; after recording, Twilio POSTs again to `/voice/start` with the recording, and the loop continues.

---

## 4. If the AI still doesn’t talk

- **Check logs** in the terminal where `./serverstart.sh` is running. You should see lines like:
  - `Voice webhook called, direction=...`
  - `LLM reply length=...`
- **If you hear "We could not connect the assistant"**  
  Twilio hit the **fallback** URL: the main URL failed (timeout, 5xx, or unreachable). Ensure ngrok is running, `NGROK_URL` is correct, and the app is listening on port 9000.
- **If you hear "One moment please" but nothing after**  
  The redirect to `/voice/start` may be failing (wrong URL or timeout). Check logs for errors when handling `/voice/start` (e.g. Groq/Sarvam errors).
- **Ensure `.env` has**  
  `GROQ_API_KEY`, `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`, `NGROK_URL`.
