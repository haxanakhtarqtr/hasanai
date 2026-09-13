import os
import json
import requests
from flask import Flask, request, Response, stream_with_context, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen-2.5-coder-32b-instruct"

def format_sse(data: str, event: str = None) -> str:
    msg = f"data: {data}\n\n"
    if event:
        msg = f"event: {event}\n{msg}"
    return msg

@app.route('/')
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
        try:
            with requests.post(OPENROUTER_URL, headers=headers, json=payload, stream=True, timeout=60) as resp:
                if resp.status_code != 200:
                    error_msg = f"OpenRouter API error: {resp.status_code} - {resp.text}"
                    yield format_sse(json.dumps({"error": error_msg}))
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
            yield format_sse(json.dumps({"error": str(e)}))

    return Response(stream_with_context(generate()), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
