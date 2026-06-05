from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import base64
import io
import docx2txt
import PyPDF2
from datetime import datetime
import os  # ← ADDED

app = Flask(__name__)
CORS(app)

# ============================================
# 🔑 YOUR GROQ API KEY - From Render Environment
# ============================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # ← CHANGED
# ============================================
# 💾 CONVERSATION MEMORY (Fixed - no repetition)
# ============================================
conversation_history = {}

def get_conversation_context(session_id, max_messages=5):
    """Get last few messages for context - NO repetition"""
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    # Only return last 3-5 messages to avoid repetition
    history = conversation_history[session_id][-max_messages:]
    
    # Format history without making AI repeat itself
    if not history:
        return ""
    
    context = "\n**Previous conversation (for context only, don't repeat):**\n"
    for entry in history:
        context += f"User: {entry['user'][:150]}\n"
        context += f"Assistant: {entry['ai'][:150]}\n"
    
    context += "\n**Important:** Use the above only for understanding context. DO NOT repeat what you already said. Answer naturally as if continuing the conversation.\n"
    return context

def save_to_history(session_id, user_msg, ai_msg):
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    conversation_history[session_id].append({
        "user": user_msg[:500],
        "ai": ai_msg[:500],
        "timestamp": datetime.now().isoformat()
    })
    
    # Keep only last 20 messages
    if len(conversation_history[session_id]) > 20:
        conversation_history[session_id] = conversation_history[session_id][-20:]

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        files = data.get('files', [])
        session_id = data.get('session_id', 'default')
        
        extracted_text = ""
        
        # Process files
        for file_info in files:
            file_name = file_info.get('name', '')
            file_data = file_info.get('data', '')
            
            if ',' in file_data:
                file_data = file_data.split(',')[1]
            
            file_bytes = base64.b64decode(file_data)
            
            try:
                if file_name.endswith('.docx'):
                    text = docx2txt.process(io.BytesIO(file_bytes))
                    extracted_text += f"\n\n📄 File: {file_name}\n{text[:5000]}"
                elif file_name.endswith('.pdf'):
                    pdf = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                    text = ""
                    for page in pdf.pages[:10]:
                        text += page.extract_text()
                    extracted_text += f"\n\n📄 File: {file_name}\n{text[:5000]}"
                elif file_name.endswith(('.txt', '.py', '.js', '.html', '.css', '.json', '.md')):
                    text = file_bytes.decode('utf-8', errors='ignore')
                    extracted_text += f"\n\n📄 File: {file_name}\n{text[:5000]}"
            except Exception as e:
                extracted_text += f"\n\nError reading {file_name}: {str(e)}"
        
        # Get conversation context
        context = get_conversation_context(session_id)
        
        # Build prompt - NO repetition instruction
        prompt = f"""{context}

**Current user message:** {user_message}

**Files content (if any):** {extracted_text if extracted_text else 'No files'}

**Instructions:**
- Respond naturally to the user's current message
- DO NOT repeat what you said in previous responses
- DO NOT summarize the conversation history
- Just answer the current question directly
- Keep responses concise unless asked for details
- Be helpful but don't over-explain things you already mentioned

**Your response (direct and natural):**"""
        
        # Call Groq API
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "You are Booster AI. Be direct, natural, and don't repeat yourself. Respond conversationally without re-introducing yourself every time."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.8,
                "max_tokens": 1500
            },
            timeout=60
        )
        
        data = response.json()
        
        if 'error' in data:
            return jsonify({'error': data['error']['message']}), 400
        
        ai_reply = data['choices'][0]['message']['content']
        
        # Save to memory
        save_to_history(session_id, user_message, ai_reply)
        
        return jsonify({'reply': ai_reply})
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear-history', methods=['POST'])
def clear_history():
    data = request.json
    session_id = data.get('session_id', 'default')
    if session_id in conversation_history:
        conversation_history[session_id] = []
        return jsonify({'status': 'History cleared'})
    return jsonify({'status': 'No history found'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'Booster AI running', 'model': 'Llama 3.3 Groq'})

# ============================================
# ✅ NEW HOME ROUTE (Fixes 404 error)
# ============================================
@app.route('/')
def home():
    return jsonify({
        'message': 'Welcome to Booster AI!',
        'status': 'running',
        'endpoints': {
            'chat': '/chat (POST)',
            'health': '/health (GET)',
            'clear-history': '/clear-history (POST)'
        },
        'docs': 'Send POST requests to /chat with {"message": "your text"}',
        'brand': 'Alpha Booster | Rwanda Creative IMG'
    })

if __name__ == '__main__':
    print("="*50)
    print("🤖 BOOSTER AI - NO REPETITION VERSION")
    print("="*50)
    print("✅ Fixed: AI won't repeat itself")
    print("✅ Natural conversations")
    print("📍 Server: http://localhost:5000")
    print("="*50)
    app.run(port=5000, debug=True, host='0.0.0.0')
