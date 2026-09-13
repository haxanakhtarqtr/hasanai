import os
import json
import time
import base64
import requests
from flask import Flask, request, Response, stream_with_context, send_from_directory, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openrouter/free"
FALLBACK_MODEL = "openrouter/free"
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}

def format_sse(data: str, event: str = None) -> str:
    msg = f"data: {data}\n\n"
    if event:
        msg = f"event: {event}\n{msg}"
    return msg

@app.route('/healthz')
def healthz():
    return jsonify({
        'status': 'ok',
        'model': MODEL,
        'timestamp': int(time.time())
    })

@app.route('/', methods=['GET'])
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    if not OPENROUTER_API_KEY:
        return Response(
            format_sse(json.dumps({"error": "OPENROUTER_API_KEY not configured"})),
            status=500,
            content_type='text/event-stream'
        )

    data = request.get_json()
    messages = data.get('messages', [])

    for msg in messages:
        if isinstance(msg.get('content'), list):
            for item in msg['content']:
                if item.get('type') == 'image_url':
                    image_url = item.get('image_url', {}).get('url', '')
                    if image_url.startswith('data:'):
                        try:
                            header, base64_data = image_url.split(',', 1)
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
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://hasanai.app",
        "X-Title": "HasanAI Chat"
    }

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True
    }

    def generate():
        tried_fallback = False
        
        def try_model(model_name):
            nonlocal tried_fallback
            payload["model"] = model_name
            try:
                with requests.post(OPENROUTER_URL, headers=headers, json=payload, stream=True, timeout=60) as resp:
                    if resp.status_code == 402 and not tried_fallback:
                        tried_fallback = True
                        yield format_sse(json.dumps({
                            "error": "Primary model unavailable. Trying a free alternative...",
                            "retry": True
                        }))
                        for chunk in try_model(FALLBACK_MODEL):
                            yield chunk
                        return
                    
                    if resp.status_code == 404:
                        yield format_sse(json.dumps({
                            "error": "Model not found. Please try again later."
                        }))
                        return
                    
                    if resp.status_code == 429:
                        yield format_sse(json.dumps({
                            "error": "Rate limit exceeded. Please wait a moment and try again."
                        }))
                        return
                    
                    if resp.status_code != 200:
                        try:
                            error_data = resp.json()
                            error_msg = error_data.get('error', {}).get('message', resp.text)
                            if 'image' in error_msg.lower() or 'vision' in error_msg.lower() or 'multimodal' in error_msg.lower():
                                error_msg = "Image processing failed. The current model may not support this image format. Try a different image or clear the image and send text only."
                        except:
                            error_msg = resp.text or f"HTTP {resp.status_code}"
                        yield format_sse(json.dumps({
                            "error": f"API Error: {error_msg}"
                        }))
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
                yield format_sse(json.dumps({
                    "error": "Request timed out. The model is taking too long to respond. Please try again."
                }))
            except requests.exceptions.ConnectionError:
                yield format_sse(json.dumps({
                    "error": "Connection error. Please check your internet connection and try again."
                }))
            except Exception as e:
                yield format_sse(json.dumps({
                    "error": f"An unexpected error occurred: {str(e)}"
                }))

        yield from try_model(MODEL)

    return Response(stream_with_context(generate()), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
