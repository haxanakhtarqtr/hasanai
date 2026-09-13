import os
import json
import time
import base64
import requests
from flask import Flask, request, Response, stream_with_context, send_from_directory, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
GOOGLE_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3.6-flash"
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}

GEMINI_MODELS = [
    {"id": "gemini-3.6-flash", "name": "Gemini 3.6 Flash (Recommended)"},
    {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
    {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash"},
]


def format_sse(data: str, event: str = None) -> str:
    msg = f"data: {data}\n\n"
    if event:
        msg = f"event: {event}\n{msg}"
    return msg


@app.route('/api/models')
def get_models():
    return jsonify({
        'models': GEMINI_MODELS,
        'default': DEFAULT_MODEL
    })


@app.route('/healthz')
def healthz():
    return jsonify({
        'status': 'ok',
        'model': DEFAULT_MODEL,
        'timestamp': int(time.time())
    })


@app.route('/', methods=['GET'])
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    if not GOOGLE_API_KEY:
        return Response(
            format_sse(json.dumps({"error": "GOOGLE_API_KEY not configured"})),
            status=500,
            content_type='text/event-stream'
        )

    data = request.get_json()
    messages = data.get('messages', [])
    selected_model = data.get('model', DEFAULT_MODEL)

    if selected_model not in [m['id'] for m in GEMINI_MODELS]:
        selected_model = DEFAULT_MODEL

    for msg in messages:
        if isinstance(msg.get('content'), list):
            for item in msg['content']:
                if item.get('type') == 'image_url':
                    image_url = item.get('image_url', {}).get('url', '')
                    if image_url.startswith('data:'):
                        try:
                            _, base64_data = image_url.split(',', 1)
                            image_bytes = base64.b64decode(base64_data)
                            if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
                                return Response(
                                    format_sse(json.dumps({"error": f"Image too large. Maximum size is {MAX_IMAGE_SIZE_BYTES // 1024 // 1024}MB"})),
                                    status=400,
                                    content_type='text/event-stream'
                                )
                        except Exception as e:
                            return Response(
                                format_sse(json.dumps({"error": f"Invalid image data: {str(e)}"})),
                                status=400,
                                content_type='text/event-stream'
                            )

    gemini_contents = []
    for msg in messages:
        role = msg.get('role', 'user')
        if role == 'system':
            role = 'user'
        content = msg.get('content', '')
        if isinstance(content, list):
            parts = []
            text_parts = []
            for item in content:
                if item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
                elif item.get('type') == 'image_url':
                    image_url = item.get('image_url', {}).get('url', '')
                    if image_url.startswith('data:'):
                        try:
                            _, base64_data = image_url.split(',', 1)
                            parts.append({
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": base64_data
                                }
                            })
                        except Exception:
                            pass
            if text_parts:
                parts.insert(0, {"text": "\n".join(text_parts)})
            if parts:
                gemini_contents.append({"role": role, "parts": parts})
        elif isinstance(content, str) and content:
            gemini_contents.append({"role": role, "parts": [{"text": content}]})

    if not gemini_contents:
        gemini_contents = [{"role": "user", "parts": [{"text": "Hello"}]}]

    def generate():
        url = f"{GOOGLE_URL_BASE}/{selected_model}:generateContent?key={GOOGLE_API_KEY}"
        payload = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048,
            }
        }

        try:
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )

            if resp.status_code != 200:
                try:
                    error_data = resp.json()
                    error_msg = error_data.get('error', {}).get('message', resp.text)
                except Exception:
                    error_msg = resp.text or f"HTTP {resp.status_code}"
                yield format_sse(json.dumps({"error": f"Google API Error: {error_msg}"}))
                return

            result = resp.json()
            text = ""
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    for part in candidate['content']['parts']:
                        if 'text' in part:
                            text += part['text']

            if text:
                yield format_sse(json.dumps({"content": text}))
            yield format_sse(json.dumps({"done": True}))

        except requests.exceptions.Timeout:
            yield format_sse(json.dumps({"error": "Request timed out. Please try again."}))
        except requests.exceptions.ConnectionError:
            yield format_sse(json.dumps({"error": "Connection error. Please try again."}))
        except Exception as e:
            yield format_sse(json.dumps({"error": f"An error occurred: {str(e)}"}))

    return Response(stream_with_context(generate()), content_type='text/event-stream')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
