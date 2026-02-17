#!/bin/bash

source venv/bin/activate
pip3 install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 9000

# In another terminal, run ngrok so Twilio can reach your app:
#   ngrok http 9000 --request-header-add "ngrok-skip-browser-warning: true"
# Without the header, Twilio's request for the Play audio URL may get an HTML page instead of the MP3.