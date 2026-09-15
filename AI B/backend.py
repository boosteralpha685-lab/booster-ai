```python
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import requests
import base64
import io
import docx2txt
import PyPDF2
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# ============================================
# 🔑 GROQ API KEY - From Render Environment
# ============================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ============================================
# 🤖 GROQ MODEL
# ============================================
GROQ_MODEL = "openai/gpt-oss-20b"

# ============================================
# 💾 CONVERSATION MEMORY
# ============================================
conversation_history = {}


def get_conversation_context(session_id, max_messages=5):
    """Get last few messages for context - NO repetition"""

    if session_id not in conversation_history:
        conversation_history[session_id] = []

    history = conversation_history[session_id][-max_messages:]

    if not history:
        return ""

    context = "\n**Previous conversation (for context only, don't repeat):**\n"

    for entry in history:
        context += f"User: {entry['user'][:150]}\n"
        context += f"Assistant: {entry['ai'][:150]}\n"

    context += (
        "\n**Important:** Use the above only for understanding context. "
        "DO NOT repeat what you already said. "
        "Answer naturally as if continuing the conversation.\n"
    )

    return context


def save_to_history(session_id, user_msg, ai_msg):
    """Save conversation to memory"""

    if session_id not in conversation_history:
        conversation_history[session_id] = []

    conversation_history[session_id].append({
        "user": user_msg[:500],
        "ai": ai_msg[:500],
        "timestamp": datetime.now().isoformat()
    })

    # Keep only the last 20 messages
    if len(conversation_history[session_id]) > 20:
        conversation_history[session_id] = (
            conversation_history[session_id][-20:]
        )


# ============================================
# 💬 CHAT
# ============================================
@app.route('/chat', methods=['POST'])
def chat():

    try:

        # Check API key
        if not GROQ_API_KEY:
            return jsonify({
                'error': 'GROQ_API_KEY is not configured on the server.'
            }), 500

        data = request.get_json(silent=True) or {}

        user_message = data.get('message', '')
        files = data.get('files', [])
        session_id = data.get('session_id', 'default')

        extracted_text = ""

        # ========================================
        # 📁 PROCESS FILES
        # ========================================
        for file_info in files:

            file_name = file_info.get('name', '')
            file_data = file_info.get('data', '')

            if not file_data:
                continue

            # Remove data URL prefix
            if ',' in file_data:
                file_data = file_data.split(',', 1)[1]

            try:
                file_bytes = base64.b64decode(file_data)
            except Exception:
                extracted_text += (
                    f"\n\n❌ Could not decode file: {file_name}"
                )
                continue

            try:

                # --------------------------------
                # DOCX
                # --------------------------------
                if file_name.lower().endswith('.docx'):

                    text = docx2txt.process(
                        io.BytesIO(file_bytes)
                    )

                    extracted_text += (
                        f"\n\n📄 File: {file_name}\n"
                        f"{text[:5000]}"
                    )

                # --------------------------------
                # PDF
                # --------------------------------
                elif file_name.lower().endswith('.pdf'):

                    pdf = PyPDF2.PdfReader(
                        io.BytesIO(file_bytes)
                    )

                    text = ""

                    for page in pdf.pages[:10]:

                        page_text = page.extract_text()

                        if page_text:
                            text += page_text

                    extracted_text += (
                        f"\n\n📄 File: {file_name}\n"
                        f"{text[:5000]}"
                    )

                # --------------------------------
                # TEXT / CODE FILES
                # --------------------------------
                elif file_name.lower().endswith(
                    (
                        '.txt',
                        '.py',
                        '.js',
                        '.html',
                        '.css',
                        '.json',
                        '.md'
                    )
                ):

                    text = file_bytes.decode(
                        'utf-8',
                        errors='ignore'
                    )

                    extracted_text += (
                        f"\n\n📄 File: {file_name}\n"
                        f"{text[:5000]}"
                    )

                else:

                    extracted_text += (
                        f"\n\n📎 File: {file_name}\n"
                        "File type is not currently supported "
                        "for text extraction."
                    )

            except Exception as e:

                extracted_text += (
                    f"\n\n❌ Error reading {file_name}: {str(e)}"
                )

        # ========================================
        # 🧠 CONVERSATION CONTEXT
        # ========================================
        context = get_conversation_context(
            session_id
        )

        # ========================================
        # 📝 BUILD PROMPT
        # ========================================
        prompt = f"""
{context}

**Current user message:**
{user_message}

**Files content (if any):**
{extracted_text if extracted_text else 'No files'}

**Instructions:**
- Respond naturally to the user's current message.
- DO NOT repeat what you said in previous responses.
- DO NOT summarize the conversation history.
- Answer the current question directly.
- Use uploaded file content when relevant.
- Keep responses concise unless asked for details.
- Be helpful without unnecessarily repeating information.

**Your response:**
"""

        # ========================================
        # 🚀 CALL GROQ API
        # ========================================
        response = requests.post(

            "https://api.groq.com/openai/v1/chat/completions",

            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },

            json={

                "model": GROQ_MODEL,

                "messages": [

                    {
                        "role": "system",
                        "content": (
                            "You are Booster AI. "
                            "Be direct, natural, and helpful. "
                            "Do not repeatedly introduce yourself. "
                            "Do not unnecessarily repeat previous answers."
                        )
                    },

                    {
                        "role": "user",
                        "content": prompt
                    }

                ],

                "temperature": 0.8,
                "max_tokens": 1500
            },

            timeout=60
        )

        # ========================================
        # 📡 READ GROQ RESPONSE
        # ========================================
        try:
            result = response.json()
        except Exception:

            return jsonify({
                'error': (
                    f'Invalid response from Groq. '
                    f'HTTP status: {response.status_code}'
                )
            }), 502

        # ========================================
        # ❌ HANDLE GROQ ERROR
        # ========================================
        if response.status_code != 200:

            error_message = (
                result.get('error', {}).get(
                    'message',
                    'Unknown Groq API error.'
                )
            )

            print(
                f"❌ Groq API error "
                f"{response.status_code}: {error_message}"
            )

            return jsonify({
                'error': error_message
            }), response.status_code

        # ========================================
        # 🤖 GET AI RESPONSE
        # ========================================
        if 'choices' not in result or not result['choices']:

            return jsonify({
                'error': 'Groq returned no AI response.'
            }), 502

        ai_reply = result['choices'][0]['message']['content']

        # ========================================
        # 💾 SAVE TO MEMORY
        # ========================================
        save_to_history(
            session_id,
            user_message,
            ai_reply
        )

        # ========================================
        # ✅ RETURN RESPONSE
        # ========================================
        return jsonify({
            'reply': ai_reply,
            'model': GROQ_MODEL
        })

    except requests.exceptions.Timeout:

        return jsonify({
            'error': 'Groq API request timed out. Please try again.'
        }), 504

    except requests.exceptions.RequestException as e:

        print(f"❌ Network error: {e}")

        return jsonify({
            'error': f'Could not connect to Groq: {str(e)}'
        }), 502

    except Exception as e:

        print(f"❌ Error: {e}")

        return jsonify({
            'error': str(e)
        }), 500


# ============================================
# 🧹 CLEAR CONVERSATION HISTORY
# ============================================
@app.route('/clear-history', methods=['POST'])
def clear_history():

    try:

        data = request.get_json(silent=True) or {}

        session_id = data.get(
            'session_id',
            'default'
        )

        if session_id in conversation_history:

            conversation_history[session_id] = []

            return jsonify({
                'status': 'History cleared'
            })

        return jsonify({
            'status': 'No history found'
        })

    except Exception as e:

        return jsonify({
            'error': str(e)
        }), 500


# ============================================
# ❤️ HEALTH CHECK
# ============================================
@app.route('/health', methods=['GET'])
def health():

    return jsonify({
        'status': 'Booster AI running',
        'model': GROQ_MODEL,
        'provider': 'Groq'
    })


# ============================================
# 🏠 HOME
# ============================================
@app.route('/', methods=['GET'])
def home():

    return jsonify({
        'message': 'Welcome to Booster AI!',
        'project': 'Booster AI',
        'status': 'running',
        'model': GROQ_MODEL,
        'endpoints': {
            'chat': '/chat (POST)',
            'health': '/health (GET)',
            'clear-history': '/clear-history (POST)'
        },
        'docs': 'Send POST requests to /chat with {"message": "your text"}',
        'brand': 'Alpha Booster | Rwanda Creative IMG'
    })


# ============================================
# 🚀 START SERVER
# ============================================
if __name__ == '__main__':

    print("=" * 50)
    print("🤖 BOOSTER AI")
    print("=" * 50)
    print("✅ Groq AI enabled")
    print(f"✅ Model: {GROQ_MODEL}")
    print("✅ Conversation memory enabled")
    print("✅ PDF/DOCX/TXT/code file reading enabled")
    print("📍 Server: http://localhost:5000")
    print("=" * 50)

    app.run(
        port=5000,
        debug=True,
        host='0.0.0.0'
    )
```
