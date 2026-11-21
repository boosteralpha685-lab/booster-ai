import os
import io
import base64
from datetime import datetime

import requests
import docx2txt
import PyPDF2
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ============================================================
# GROQ CONFIGURATION
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

# Groq's currently listed production model.
# You can override this in your hosting environment if needed.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b").strip()

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
GROQ_TIMEOUT = int(os.environ.get("GROQ_TIMEOUT", "60"))

# ============================================================
# CONVERSATION MEMORY
# NOTE: This is in-memory only; it resets when the server restarts.
# ============================================================
conversation_history = {}
MAX_HISTORY_MESSAGES = 20
MAX_CONTEXT_MESSAGES = 5
MAX_FILE_TEXT_CHARS = 12000
MAX_PROMPT_CHARS = 24000


def get_conversation_context(session_id, max_messages=MAX_CONTEXT_MESSAGES):
    history = conversation_history.get(session_id, [])[-max_messages:]
    if not history:
        return ""

    parts = ["\nPrevious conversation (context only):"]
    for entry in history:
        parts.append(f"User: {entry['user'][:300]}")
        parts.append(f"Assistant: {entry['ai'][:300]}")

    parts.append(
        "Use this only to understand the conversation. "
        "Do not repeat previous answers unless necessary."
    )
    return "\n".join(parts)


def save_to_history(session_id, user_msg, ai_msg):
    history = conversation_history.setdefault(session_id, [])
    history.append({
        "user": str(user_msg)[:1000],
        "ai": str(ai_msg)[:1500],
        "timestamp": datetime.now().isoformat()
    })
    conversation_history[session_id] = history[-MAX_HISTORY_MESSAGES:]


def extract_uploaded_files(files):
    """Extract text from supported uploaded documents safely."""
    extracted_parts = []

    if not isinstance(files, list):
        return ""

    total_chars = 0

    for file_info in files:
        if not isinstance(file_info, dict):
            continue

        file_name = str(file_info.get("name", "uploaded_file"))
        file_data = file_info.get("data", "")

        if not isinstance(file_data, str) or not file_data:
            continue

        # Accept both raw base64 and data URLs.
        if "," in file_data and file_data.lstrip().startswith("data:"):
            file_data = file_data.split(",", 1)[1]

        try:
            file_bytes = base64.b64decode(file_data, validate=False)
        except Exception:
            extracted_parts.append(f"\nCould not decode file: {file_name}")
            continue

        try:
            lower_name = file_name.lower()
