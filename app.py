import os
import json
import time
import base64
import requests
from flask import Flask, request, Response, stream_with_context, send_from_directory, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Providers
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')

# URLs
GOOGLE_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

DEFAULT_MODEL = "gemini-3.6-flash"
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}

MODELS = [
    # Google Gemini
    {"id": "gemini-3.6-flash", "name": "Gemini 3.6 Flash (Recommended)", "provider": "google"},
    {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "provider": "google"},
    {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash", "provider": "google"},
    # Groq
    {"id": "groq/llama-3.1-8b-instant", "name": "Llama 3.1 8B (Groq)", "provider": "groq"},
    {"id": "groq/llama-3.2-3b-preview", "name": "Llama 3.2 3B (Groq)", "provider": "groq"},
    {"id": "groq/gemma2-9b-it", "name": "Gemma 2 9B (Groq)", "provider": "groq"},
    {"id": "groq/mixtral-8x7b-32768", "name": "Mixtral 8x7B (Groq)", "provider": "groq"},
    # OpenRouter
    {"id": "openrouter/free", "name": "Free Model Router (OpenRouter)", "provider": "openrouter"},
    {"id": "openrouter/auto", "name": "Auto Router (OpenRouter)", "provider": "openrouter"},
    # Mistral
    {"id": "mistral/mistral-small-latest", "name": "Mistral Small", "provider": "mistral"},
    {"id": "mistral/mistral-large-latest", "name": "Mistral Large", "provider": "mistral"},
    {"id": "mistral/ministral-8b-latest", "name": "Ministral 8B", "provider": "mistral"},
]

PROVIDER_FALLBACK_ORDER = ["google", "groq", "openrouter", "mistral"]


def format_sse(data: str, event: str = None) -> str:
    msg = f"data: {data}\n\n"
    if event:
        msg = f"event: {event}\n{msg}"
    return msg


def get_provider_for_model(model_id: str) -> str:
    for m in MODELS:
        if m['id'] == model_id:
            return m['provider']
    return 'google'


def get_next_provider(current_provider: str) -> str | None:
    try:
        idx = PROVIDER_FALLBACK_ORDER.index(current_provider)
        for next_idx in range(idx + 1, len(PROVIDER_FALLBACK_ORDER)):
            next_provider = PROVIDER_FALLBACK_ORDER[next_idx]
            if next_provider == 'google' and GOOGLE_API_KEY:
                return next_provider
            if next_provider == 'groq' and GROQ_API_KEY:
                return next_provider
            if next_provider == 'openrouter' and OPENROUTER_API_KEY:
                return next_provider
            if next_provider == 'mistral' and MISTRAL_API_KEY:
                return next_provider
    except ValueError:
        pass
    return None


@app.route('/api/models')
def get_models():
    return jsonify({
        'models': MODELS,
        'default': DEFAULT_MODEL
    })


@app.route('/healthz')
def healthz():
    return jsonify({
        'status': 'ok',
        'model': DEFAULT_MODEL,
        'providers': {
            'google': bool(GOOGLE_API_KEY),
            'groq': bool(GROQ_API_KEY),
            'openrouter': bool(OPENROUTER_API_KEY),
            'mistral': bool(MISTRAL_API_KEY),
        },
        'timestamp': int(time.time())
    })


@app.route('/', methods=['GET'])
def index():
    return send_from_directory('.', 'index.html')


def call_google_chat(messages, model, has_image=False):
    url = f"{GOOGLE_URL_BASE}/{model}:generateContent?key={GOOGLE_API_KEY}"
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
                            image_bytes = base64.b64decode(base64_data)
                            if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
                                yield format_sse(json.dumps({"error": "Image too large. Maximum size is 20MB"}))
                                return
                            parts.append({
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": base64_data
                                }
                            })
                        except Exception as e:
                            yield format_sse(json.dumps({"error": f"Invalid image data: {str(e)}"}))
                            return
            if text_parts:
                parts.insert(0, {"text": "\n".join(text_parts)})
            if parts:
                gemini_contents.append({"role": role, "parts": parts})
        elif isinstance(content, str) and content:
            gemini_contents.append({"role": role, "parts": [{"text": content}]})

    if not gemini_contents:
        gemini_contents = [{"role": "user", "parts": [{"text": "Hello"}]}]

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
    except Exception as e:
        yield format_sse(json.dumps({"error": f"Google API error: {str(e)}"}))


def call_groq_chat(messages, model):
    url = GROQ_URL
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model.replace("groq/", ""),
        "messages": messages,
        "stream": True,
        "max_tokens": 2048,
        "temperature": 0.7
    }
    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as resp:
            if resp.status_code != 200:
                try:
                    error_data = resp.json()
                    error_msg = error_data.get('error', {}).get('message', resp.text)
                except Exception:
                    error_msg = resp.text or f"HTTP {resp.status_code}"
                yield format_sse(json.dumps({"error": f"Groq API Error: {error_msg}"}))
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
    except Exception as e:
        yield format_sse(json.dumps({"error": f"Groq API error: {str(e)}"}))


def call_openrouter_chat(messages, model):
    url = OPENROUTER_URL
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://hasanai.app",
        "X-Title": "HasanAI Chat"
    }
    payload = {
        "model": model.replace("openrouter/", ""),
        "messages": messages,
        "stream": True,
        "max_tokens": 2048,
        "temperature": 0.7
    }
    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as resp:
            if resp.status_code == 402:
                yield format_sse(json.dumps({"error": "OpenRouter credits exhausted.", "retry": True}))
                return
            if resp.status_code == 429:
                yield format_sse(json.dumps({"error": "OpenRouter rate limit exceeded.", "retry": True}))
                return
            if resp.status_code != 200:
                try:
                    error_data = resp.json()
                    error_msg = error_data.get('error', {}).get('message', resp.text)
                except Exception:
                    error_msg = resp.text or f"HTTP {resp.status_code}"
                yield format_sse(json.dumps({"error": f"OpenRouter API Error: {error_msg}"}))
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
    except Exception as e:
        yield format_sse(json.dumps({"error": f"OpenRouter API error: {str(e)}"}))


def call_mistral_chat(messages, model):
    url = MISTRAL_URL
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model.replace("mistral/", ""),
        "messages": messages,
        "stream": True,
        "max_tokens": 2048,
        "temperature": 0.7
    }
    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as resp:
            if resp.status_code != 200:
                try:
                    error_data = resp.json()
                    error_msg = error_data.get('error', {}).get('message', resp.text)
                except Exception:
                    error_msg = resp.text or f"HTTP {resp.status_code}"
                yield format_sse(json.dumps({"error": f"Mistral API Error: {error_msg}"}))
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
    except Exception as e:
        yield format_sse(json.dumps({"error": f"Mistral API error: {str(e)}"}))


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    messages = data.get('messages', [])
    selected_model = data.get('model', DEFAULT_MODEL)
    provider = get_provider_for_model(selected_model)

    has_image = False
    for msg in messages:
        if isinstance(msg.get('content'), list):
            for item in msg['content']:
                if item.get('type') == 'image_url':
                    has_image = True
                    break

    # Check if selected provider is available
    provider_available = {
        'google': bool(GOOGLE_API_KEY),
        'groq': bool(GROQ_API_KEY),
        'openrouter': bool(OPENROUTER_API_KEY),
        'mistral': bool(MISTRAL_API_KEY),
    }

    if not provider_available.get(provider, False):
        # Try fallback providers
        fallback_models = {
            'google': 'gemini-3.6-flash',
            'groq': 'groq/llama-3.1-8b-instant',
            'openrouter': 'openrouter/free',
            'mistral': 'mistral/mistral-small-latest',
        }
        
        for fallback_provider in PROVIDER_FALLBACK_ORDER:
            if provider_available.get(fallback_provider, False):
                selected_model = fallback_models.get(fallback_provider, DEFAULT_MODEL)
                provider = fallback_provider
                yield format_sse(json.dumps({"error": f"Primary provider unavailable. Switching to {provider}..."}))
                break
        else:
            return Response(
                format_sse(json.dumps({"error": "No API providers configured. Please add at least one API key."})),
                status=500,
                content_type='text/event-stream'
            )

    # Convert messages to provider format
    if provider == 'google':
        return Response(stream_with_context(call_google_chat(messages, selected_model, has_image)), content_type='text/event-stream')
    elif provider == 'groq':
        return Response(stream_with_context(call_groq_chat(messages, selected_model)), content_type='text/event-stream')
    elif provider == 'openrouter':
        return Response(stream_with_context(call_openrouter_chat(messages, selected_model)), content_type='text/event-stream')
    elif provider == 'mistral':
        return Response(stream_with_context(call_mistral_chat(messages, selected_model)), content_type='text/event-stream')
    else:
        return Response(
            format_sse(json.dumps({"error": "Unknown provider"})),
            status=400,
            content_type='text/event-stream'
        )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
