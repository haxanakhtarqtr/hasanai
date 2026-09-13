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
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=" + str(GOOGLE_API_KEY or '')
MODEL = "openrouter/free"
FALLBACK_CHAIN = [
    "openrouter/free",
    "google/gemini-2.5-flash:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-2-9b-it:free",
    "nvidia/llama-3.1-nemotron-8b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "qwen/qwen3-coder:free",
    "poolside/laguna-xs-2.1:free",
]
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}
FREE_MODELS = [
    {"id": "openrouter/free", "name": "Free Model Router"},
    {"id": "openai/gpt-oss-20b:free", "name": "GPT-OSS 20B"},
    {"id": "openai/gpt-oss-120b:free", "name": "GPT-OSS 120B"},
    {"id": "meta-llama/llama-3.3-70b-instruct:free", "name": "Llama 3.3 70B"},
    {"id": "meta-llama/llama-3.1-8b-instruct:free", "name": "Llama 3.1 8B"},
    {"id": "meta-llama/llama-3.2-3b-instruct:free", "name": "Llama 3.2 3B"},
    {"id": "google/gemma-4-31b-it:free", "name": "Gemma 4 31B"},
    {"id": "google/gemma-4-26b-a4b-it:free", "name": "Gemma 4 26B A4B"},
    {"id": "google/gemma-2-9b-it:free", "name": "Gemma 2 9B"},
    {"id": "nvidia/nemotron-3-ultra-550b-a55b:free", "name": "Nemotron 3 Ultra"},
    {"id": "nvidia/nemotron-3-super-120b-a12b:free", "name": "Nemotron 3 Super"},
    {"id": "nvidia/nemotron-3-nano-30b-a3b:free", "name": "Nemotron 3 Nano"},
    {"id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "name": "Nemotron Nano Omni"},
    {"id": "nvidia/nemotron-3.5-lightning:free", "name": "Nemotron 3.5 Lightning"},
    {"id": "mistralai/mistral-7b-instruct:free", "name": "Mistral 7B"},
    {"id": "microsoft/phi-3-mini-128k-instruct:free", "name": "Phi-3 Mini"},
    {"id": "cohere/north-mini-code:free", "name": "North Mini Code"},
    {"id": "poolside/laguna-m.1:free", "name": "Laguna M.1"},
    {"id": "poolside/laguna-xs-2.1:free", "name": "Laguna XS.1"},
    {"id": "poolside/laguna-s-2.1:free", "name": "Laguna S.1"},
    {"id": "qwen/qwen3-coder:free", "name": "Qwen3 Coder"},
    {"id": "qwen/qwen3-next-80b-a3b-instruct:free", "name": "Qwen3 Next 80B"},
    {"id": "z-ai/glm-5.2:free", "name": "GLM 5.2"},
    {"id": "minimax/minimax-m3:free", "name": "MiniMax M3"},
    {"id": "minimax/minimax-m2.7:free", "name": "MiniMax M2.7"},
    {"id": "huggingfaceh4/zephyr-7b-beta:free", "name": "Zephyr 7B"},
]

def format_sse(data: str, event: str = None) -> str:
    msg = f"data: {data}\n\n"
    if event:
        msg = f"event: {event}\n{msg}"
    return msg

@app.route('/api/models')
def get_models():
    models = FREE_MODELS + [
        {"id": "google/gemini-2.5-flash:free", "name": "Gemini 2.5 Flash (Google)"},
        {"id": "google/gemini-1.5-flash:free", "name": "Gemini 1.5 Flash (Google)"},
    ]
    return jsonify({
        'models': models,
        'default': MODEL
    })

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
    selected_model = data.get('model', MODEL)

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
        "model": selected_model,
        "messages": messages,
        "stream": True
    }

    def generate():
        tried_fallback = False
        fallback_index = [0]
        
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
                        for chunk in try_next_free_model(fallback_index):
                            yield chunk
                        return
                    
                    if resp.status_code == 429 and not tried_fallback:
                        tried_fallback = True
                        yield format_sse(json.dumps({
                            "error": "Rate limit exceeded. Trying another free model...",
                            "retry": True
                        }))
                        for chunk in try_next_free_model(fallback_index):
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

        def try_next_free_model(index_ref):
            for i in range(1, len(FALLBACK_CHAIN)):
                next_index = (index_ref[0] + i) % len(FALLBACK_CHAIN)
                if next_index == 0 and i > 0:
                    continue
                index_ref[0] = next_index
                yield format_sse(json.dumps({
                    "error": f"Trying alternative model: {FALLBACK_CHAIN[next_index].split('/')[-1]}"
                }))
                for chunk in try_model(FALLBACK_CHAIN[next_index]):
                    yield chunk
                return
            yield format_sse(json.dumps({
                "error": "All models are currently unavailable. Please try again later."
            }))

        yield from try_model(selected_model)

    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route('/api/chat/google', methods=['POST'])
def chat_google():
    if not GOOGLE_API_KEY:
        return Response(
            format_sse(json.dumps({"error": "GOOGLE_API_KEY not configured"})),
            status=500,
            content_type='text/event-stream'
        )

    data = request.get_json()
    messages = data.get('messages', [])
    
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
        payload = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048,
            }
        }
        
        try:
            resp = requests.post(
                GOOGLE_URL,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )
            
            if resp.status_code != 200:
                try:
                    error_data = resp.json()
                    error_msg = error_data.get('error', {}).get('message', resp.text)
                except:
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
