import os
import json
import time
import base64
import requests
from flask import Flask, request, Response, stream_with_context, send_from_directory, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get('API_KEY', 'atr_VsYOuvA90HXbeSqMsh_PKR2OWvVuJebe')
API_URL = os.environ.get('API_URL', 'https://api.atria-asi.ai/v1/chat/completions')
DEFAULT_MODEL = os.environ.get('DEFAULT_MODEL', 'Atria-Dawn-Preview')
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}


def format_sse(data: str, event: str = None) -> str:
    msg = f"data: {data}\n\n"
    if event:
        msg = f"event: {event}\n{msg}"
    return msg


@app.route('/api/models')
def get_models():
    return jsonify({
        'models': [{"id": DEFAULT_MODEL, "name": "Atria Dawn Preview"}],
        'default': DEFAULT_MODEL
    })


@app.route('/healthz')
def healthz():
    return jsonify({
        'status': 'ok',
        'model': DEFAULT_MODEL,
        'provider': 'atria',
        'configured': bool(API_KEY and API_URL),
        'timestamp': int(time.time())
    })


@app.route('/', methods=['GET'])
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    if not API_KEY or not API_URL:
        return Response(
            format_sse(json.dumps({"error": "Provider not configured yet."})),
            status=500,
            content_type='text/event-stream'
        )

    data = request.get_json()
    messages = data.get('messages', [])
    selected_model = data.get('model', DEFAULT_MODEL)

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

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": selected_model,
        "messages": messages,
        "stream": True
    }

    try:
        with requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=60) as resp:
            if resp.status_code != 200:
                try:
                    error_data = resp.json()
                    error_msg = error_data.get('message') or error_data.get('error', {}).get('message') or resp.text
                except Exception:
                    error_msg = resp.text or f"HTTP {resp.status_code}"
                yield format_sse(json.dumps({"error": f"Atria API Error: {error_msg}"}))
                return

            for line in resp.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        line = line[6:]
                    if line.strip() == '[DONE]':
                        yield format_sse(json.dumps({"done": True}))
                        return
                    try:
                        chunk = json.loads(line)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                yield format_sse(json.dumps({"content": content}))
                    except json.JSONDecodeError:
                        continue
    except requests.exceptions.Timeout:
        yield format_sse(json.dumps({"error": "Request timed out. Please try again."}))
    except requests.exceptions.ConnectionError:
        yield format_sse(json.dumps({"error": "Connection error. Please try again."}))
    except Exception as e:
        yield format_sse(json.dumps({"error": f"An error occurred: {str(e)}"}))


@app.route('/api/chat-debug', methods=['POST'])
def chat_debug():
    data = request.get_json()
    messages = data.get('messages', [])
    selected_model = data.get('model', DEFAULT_MODEL)
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": selected_model,
        "messages": messages,
        "stream": False
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        return jsonify({
            'status_code': resp.status_code,
            'headers': dict(resp.headers),
            'text': resp.text[:1000]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
