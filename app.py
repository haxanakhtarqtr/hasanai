import jwt
import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, Conversation, Message, Setting
import os
import json
import time
import base64
import requests
from flask import Flask, request, Response, stream_with_context, send_from_directory, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'hasanai-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///hasanai.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    db.create_all()

API_KEY = os.environ.get('API_KEY', 'sk_4483bb8b0eb01bac0667948aea29eeea')
API_URL = os.environ.get('API_URL', 'https://api.inceptionlabs.ai/v1/chat/completions')
DEFAULT_MODEL = os.environ.get('DEFAULT_MODEL', 'mercury-2.5')
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}


def format_sse(data: str, event: str = None) -> str:
    msg = f"data: {data}\n\n"
    if event:
        msg = f"event: {event}\n{msg}"
    return msg


def token_required(f):
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Token is missing'}), 401
        token = auth_header.split(' ')[1]
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
            if not current_user:
                return jsonify({'error': 'User not found'}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        return f(current_user, *args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper


def optional_token(f):
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        user = None
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            try:
                data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
                user = User.query.get(data['user_id'])
            except Exception:
                pass
        return f(user, *args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper


@app.route('/api/models')
def get_models():
    return jsonify({
        'models': [{"id": DEFAULT_MODEL, "name": "Mercury 2.5"}],
        'default': DEFAULT_MODEL
    })


@app.route('/healthz')
def healthz():
    return jsonify({
        'status': 'ok',
        'model': DEFAULT_MODEL,
        'provider': 'inceptionlabs',
        'configured': bool(API_KEY and API_URL),
        'timestamp': int(time.time())
    })


@app.route('/', methods=['GET'])
def index():
    resp = send_from_directory('.', 'index.html')
    # Never let the browser or an intermediary hold a stale app shell; a cached
    # index.html hides new fixes from users even after the file on disk changes.
    resp.headers['Cache-Control'] = 'no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    return resp


@app.route('/manifest.json')
def manifest():
    return send_from_directory('.', 'manifest.json')


@app.route('/sw.js')
def service_worker():
    resp = send_from_directory('.', 'sw.js')
    # Must always be revalidated so service worker updates are picked up.
    resp.headers['Cache-Control'] = 'no-store, must-revalidate'
    return resp


@app.route('/api/chat', methods=['POST'])
@optional_token
def chat(current_user):
    if not API_KEY or not API_URL:
        return Response(
            format_sse(json.dumps({"error": "Provider not configured yet."})),
            status=500,
            content_type='text/event-stream'
        )

    data = request.get_json()
    messages = data.get('messages', [])
    selected_model = data.get('model', DEFAULT_MODEL)
    api_url = data.get('apiUrl') or API_URL
    api_key = data.get('apiKey') or API_KEY

    if not api_key or not api_url:
        return Response(
            format_sse(json.dumps({"error": "Provider not configured yet."})),
            status=500,
            content_type='text/event-stream'
        )

    try:
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
    except Exception as e:
        return Response(
            format_sse(json.dumps({"error": f"Failed to process request: {str(e)}"})),
            status=400,
            content_type='text/event-stream'
        )

    def to_text_messages(msgs):
        api_messages = []
        for msg in msgs:
            content = msg.get('content')
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                text = ' '.join(text_parts).strip()
                api_messages.append({'role': msg.get('role', 'user'), 'content': text})
            else:
                api_messages.append(msg)
        return api_messages

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": selected_model,
        "messages": to_text_messages(messages),
        "stream": True
    }

    def generate():
        try:
            with requests.post(api_url, headers=headers, json=payload, stream=True, timeout=60) as resp:
                if resp.status_code != 200:
                    try:
                        error_data = resp.json()
                        error_msg = error_data.get('message') or error_data.get('error', {}).get('message') or resp.text
                    except Exception:
                        error_msg = resp.text or f"HTTP {resp.status_code}"
                    yield format_sse(json.dumps({"error": f"API Error: {error_msg}"}))
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

    try:
        return Response(stream_with_context(generate()), content_type='text/event-stream')
    except Exception as e:
        return Response(
            format_sse(json.dumps({"error": f"Failed to start streaming: {str(e)}"})),
            status=200,
            content_type='text/event-stream'
        )


@app.route('/api/chat-debug', methods=['POST'])
def chat_debug():
    data = request.get_json()
    messages = data.get('messages', [])
    selected_model = data.get('model', DEFAULT_MODEL)
    api_url = data.get('apiUrl') or API_URL
    api_key = data.get('apiKey') or API_KEY
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": selected_model,
        "messages": to_text_messages(messages),
        "stream": False
    }
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
        return jsonify({
            'status_code': resp.status_code,
            'headers': dict(resp.headers),
            'text': resp.text[:1000]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/auth/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    name = data.get('name', '').strip()
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 400
    user = User(
        email=email,
        password_hash=generate_password_hash(password),
        name=name or email.split('@')[0]
    )
    db.session.add(user)
    db.session.commit()
    token = jwt.encode({
        'user_id': user.id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=30)
    }, app.config['SECRET_KEY'], algorithm='HS256')
    return jsonify({
        'token': token,
        'user': {'id': user.id, 'email': user.email, 'name': user.name}
    })


@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({'error': 'Invalid email or password'}), 401
    token = jwt.encode({
        'user_id': user.id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=30)
    }, app.config['SECRET_KEY'], algorithm='HS256')
    return jsonify({
        'token': token,
        'user': {'id': user.id, 'email': user.email, 'name': user.name}
    })


@app.route('/api/auth/logout', methods=['POST'])
@token_required
def logout(current_user):
    return jsonify({'message': 'Logged out successfully'})


@app.route('/api/user/me', methods=['GET'])
@token_required
def get_current_user(current_user):
    return jsonify({'user': {'id': current_user.id, 'email': current_user.email, 'name': current_user.name}})


@app.route('/api/conversations', methods=['GET'])
@token_required
def get_conversations(current_user):
    convs = Conversation.query.filter_by(user_id=current_user.id).order_by(Conversation.updated_at.desc()).all()
    return jsonify({'conversations': [{
        'id': c.id,
        'title': c.title,
        'created_at': c.created_at.isoformat(),
        'updated_at': c.updated_at.isoformat(),
        'messages': [{'role': m.role, 'content': m.content, 'created_at': m.created_at.isoformat()} for m in c.messages]
    } for c in convs]})


def _message_content_to_text(content):
    """Convert a message content (which may be a list of parts or a plain string)
    into a single text string for storage."""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text' and item.get('text'):
                parts.append(str(item['text']))
        return ' '.join(parts).strip()
    return str(content) if content else ''


def _replace_messages(conversation, messages):
    Message.query.filter_by(conversation_id=conversation.id).delete()
    if isinstance(messages, list):
        for m in messages:
            if isinstance(m, dict) and m.get('role') in ('user', 'assistant', 'system'):
                content = _message_content_to_text(m.get('content'))
                db.session.add(Message(conversation_id=conversation.id, role=str(m.get('role')), content=content))


@app.route('/api/conversations', methods=['POST'])
@token_required
def create_conversation(current_user):
    data = request.get_json() or {}
    conv = Conversation(user_id=current_user.id, title=data.get('title') or 'New Chat')
    db.session.add(conv)
    db.session.flush()
    _replace_messages(conv, data.get('messages'))
    db.session.commit()
    return jsonify({'conversation': {'id': conv.id, 'title': conv.title, 'created_at': conv.created_at.isoformat(), 'updated_at': conv.updated_at.isoformat(), 'messages': [{'role': m.role, 'content': m.content, 'created_at': m.created_at.isoformat()} for m in conv.messages]}})


@app.route('/api/conversations/<int:conv_id>', methods=['PUT'])
@token_required
def update_conversation(current_user, conv_id):
    conv = Conversation.query.filter_by(id=conv_id, user_id=current_user.id).first_or_404()
    data = request.get_json() or {}
    if 'title' in data and data['title']:
        conv.title = str(data['title'])
    if 'messages' in data:
        _replace_messages(conv, data['messages'])
    conv.updated_at = datetime.datetime.utcnow()
    db.session.commit()
    return jsonify({'conversation': {'id': conv.id, 'title': conv.title, 'created_at': conv.created_at.isoformat(), 'updated_at': conv.updated_at.isoformat(), 'messages': [{'role': m.role, 'content': m.content, 'created_at': m.created_at.isoformat()} for m in conv.messages]}})


@app.route('/api/conversations/<int:conv_id>', methods=['DELETE'])
@token_required
def delete_conversation(current_user, conv_id):
    conv = Conversation.query.filter_by(id=conv_id, user_id=current_user.id).first_or_404()
    db.session.delete(conv)
    db.session.commit()
    return jsonify({'message': 'Deleted'})


@app.route('/api/settings', methods=['GET'])
@token_required
def get_settings(current_user):
    settings = Setting.query.filter_by(user_id=current_user.id).all()
    return jsonify({'settings': {s.key: s.value for s in settings}})


@app.route('/api/settings', methods=['POST'])
@token_required
def save_settings(current_user):
    data = request.get_json()
    for key, value in (data or {}).items():
        setting = Setting.query.filter_by(user_id=current_user.id, key=key).first()
        if setting:
            setting.value = str(value)
        else:
            setting = Setting(user_id=current_user.id, key=key, value=str(value))
            db.session.add(setting)
    db.session.commit()
    return jsonify({'message': 'Settings saved'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)