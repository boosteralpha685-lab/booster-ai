import os
import io
import base64
from datetime import datetime

import requests
import docx2txt
import PyPDF2
from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)
CORS(app)

# Prevent unexpectedly large uploads. Adjust in Render if needed.
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("MAX_CONTENT_LENGTH", str(20 * 1024 * 1024))
)

# ============================================================
# GROQ CONFIGURATION
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b").strip()
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.8-27b").strip()
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT = int(os.environ.get("GROQ_TIMEOUT", "60"))

# ============================================================
# CONVERSATION MEMORY
# NOTE: In-memory only. It resets when Render restarts the app.
# ============================================================
conversation_history = {}

MAX_HISTORY_MESSAGES = 20
MAX_CONTEXT_MESSAGES = 5
MAX_FILE_TEXT_CHARS = 12000
MAX_PROMPT_CHARS = 24000


def get_conversation_context(session_id, max_messages=MAX_CONTEXT_MESSAGES):
    """Return recent conversation turns as chat messages."""
    history = conversation_history.get(session_id, [])[-max_messages:]
    messages = []

    for entry in history:
        messages.append({"role": "user", "content": entry["user"]})
        messages.append({"role": "assistant", "content": entry["ai"]})

    return messages


def save_to_history(session_id, user_msg, ai_msg):
    """Save one completed exchange in the in-memory conversation history."""
    history = conversation_history.setdefault(session_id, [])
    history.append({
        "user": str(user_msg)[:1000],
        "ai": str(ai_msg)[:1500],
        "timestamp": datetime.now().isoformat()
    })
    conversation_history[session_id] = history[-MAX_HISTORY_MESSAGES:]


def extract_uploaded_files(files):
    """Extract text from supported uploaded documents safely."""
    if not isinstance(files, list):
        return ""

    extracted_parts = []
    total_chars = 0

    for file_info in files:
        if not isinstance(file_info, dict):
            continue

        file_name = str(file_info.get("name", "uploaded_file"))
        file_data = file_info.get("data", "")

        if not isinstance(file_data, str) or not file_data:
            continue

        # Accept raw base64 or a data URL such as data:application/pdf;base64,...
        if file_data.lstrip().startswith("data:") and "," in file_data:
            file_data = file_data.split(",", 1)[1]

        try:
            file_bytes = base64.b64decode(file_data, validate=False)
        except Exception:
            extracted_parts.append(f"\nCould not decode file: {file_name}")
            continue

        try:
            lower_name = file_name.lower()
            text = ""

            if lower_name.endswith(".docx"):
                text = docx2txt.process(io.BytesIO(file_bytes))

            elif lower_name.endswith(".pdf"):
                pdf = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                page_texts = []

                for page in pdf.pages[:10]:
                    page_text = page.extract_text() or ""
                    if page_text:
                        page_texts.append(page_text)

                text = "\n".join(page_texts)

            elif lower_name.endswith((
                ".txt", ".py", ".js", ".html", ".css", ".json", ".md",
                ".xml", ".csv", ".sql", ".java", ".kt"
            )):
                text = file_bytes.decode("utf-8", errors="ignore")

            else:
                text = (
                    "File type is not currently supported for text extraction."
                )

            remaining = MAX_FILE_TEXT_CHARS - total_chars
            if remaining <= 0:
                extracted_parts.append(
                    "\nAdditional file content omitted because the upload "
                    "text limit was reached."
                )
                break

            text = (text or "")[:remaining]
            total_chars += len(text)
            extracted_parts.append(f"\n\nFile: {file_name}\n{text}")

        except Exception as exc:
            extracted_parts.append(
                f"\n\nError reading {file_name}: {str(exc)}"
            )

    return "\n".join(extracted_parts)


# ============================================================
# CHAT
# ============================================================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        if not GROQ_API_KEY:
            return jsonify({
                "error": "GROQ_API_KEY is not configured on the server."
            }), 500

        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        user_message = data.get("message", "")
        if not isinstance(user_message, str):
            return jsonify({"error": "The message must be text."}), 400

        files = data.get("files", [])
        session_id = str(data.get("session_id") or "default")[:128]

        extracted_text = extract_uploaded_files(files)

        prompt = user_message.strip()
        if extracted_text:
            prompt += (
                "\n\nUploaded file content (use it when relevant):\n"
                + extracted_text
            )

        if len(prompt) > MAX_PROMPT_CHARS:
            prompt = prompt[:MAX_PROMPT_CHARS] + "\n\n[Content truncated.]"

        messages = [{
            "role": "system",
            "content": (
                "You are Booster AI. Be direct, natural, and helpful. "
                "Do not repeatedly introduce yourself. Use conversation "
                "history only when it helps answer the current message. "
                "When files are provided, use their contents when relevant. "
                "If the file does not contain an answer, say so rather than "
                "inventing details."
            )
        }]

        messages.extend(get_conversation_context(session_id))
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            GROQ_CHAT_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": 0.8,
                "max_completion_tokens": 1500
            },
            timeout=GROQ_TIMEOUT
        )

        try:
            result = response.json()
        except ValueError:
            return jsonify({
                "error": (
                    "Groq returned an invalid response. "
                    f"HTTP status: {response.status_code}"
                )
            }), 502

        if response.status_code != 200:
            error_data = result.get("error", {}) if isinstance(result, dict) else {}
            error_message = (
                error_data.get("message", "Unknown Groq API error.")
                if isinstance(error_data, dict)
                else "Unknown Groq API error."
            )
            print(f"Groq API error {response.status_code}: {error_message}")
            return jsonify({"error": error_message}), 502

        choices = result.get("choices") if isinstance(result, dict) else None
        if not choices:
            return jsonify({"error": "Groq returned no AI response."}), 502

        ai_reply = (
            choices[0].get("message", {}).get("content")
            if isinstance(choices[0], dict)
            else None
        )

        if not isinstance(ai_reply, str) or not ai_reply.strip():
            return jsonify({"error": "Groq returned an empty AI response."}), 502

        save_to_history(session_id, user_message, ai_reply)

        return jsonify({
            "reply": ai_reply,
            "model": GROQ_MODEL
        }), 200

    except requests.exceptions.Timeout:
        return jsonify({
            "error": "Groq API request timed out. Please try again."
        }), 504

    except requests.exceptions.RequestException as exc:
        print(f"Network error while connecting to Groq: {exc}")
        return jsonify({
            "error": "Could not connect to Groq. Please try again."
        }), 502

    except Exception as exc:
        print(f"Unexpected /chat error: {exc}")
        return jsonify({"error": "An unexpected server error occurred."}), 500


# ============================================================
# IMAGE ANALYSIS
# Uses a separate vision-capable model; the normal chat model
# remains configured by GROQ_MODEL.
# POST /analyze-image
# JSON: {"image": "data:image/jpeg;base64,...", "question": "..."}
# Also accepts "image_data" instead of "image".
# ============================================================
@app.route("/analyze-image", methods=["POST"])
@app.route("/image-analyze", methods=["POST"])
def analyze_image():
    try:
        if not GROQ_API_KEY:
            return jsonify({
                "error": "GROQ_API_KEY is not configured on the server."
            }), 500

        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        image_value = data.get("image") or data.get("image_data") or ""
        question = data.get("question") or data.get("message") or (
            "Describe this image in detail. Identify visible objects, text, "
            "layout, and useful observations. Do not guess hidden details."
        )

        if not isinstance(image_value, str) or not image_value.strip():
            return jsonify({
                "error": "Please provide an image in base64 or data-URL format."
            }), 400

        image_value = image_value.strip()
        mime_type = str(data.get("mime_type") or "image/jpeg").lower().strip()

        if image_value.startswith("data:"):
            try:
                header, image_base64 = image_value.split(",", 1)
                mime_type = header[5:].split(";", 1)[0].lower()
                image_value = image_base64
            except ValueError:
                return jsonify({"error": "Invalid image data URL."}), 400

        allowed_mime_types = {
            "image/jpeg", "image/jpg", "image/png", "image/webp"
        }
        if mime_type == "image/jpg":
            mime_type = "image/jpeg"

        if mime_type not in allowed_mime_types:
            return jsonify({
                "error": "Unsupported image type. Use JPG, PNG, or WEBP."
            }), 400

        try:
            image_bytes = base64.b64decode(image_value, validate=True)
        except Exception:
            return jsonify({"error": "The image is not valid base64 data."}), 400

        if not image_bytes:
            return jsonify({"error": "The uploaded image is empty."}), 400

        # Keep image payloads below the vision provider's documented limit.
        if len(image_bytes) > 15 * 1024 * 1024:
            return jsonify({
                "error": "Image is too large. Please upload an image under 15 MB."
            }), 413

        image_data_url = (
            f"data:{mime_type};base64,"
            + base64.b64encode(image_bytes).decode("utf-8")
        )

        response = requests.post(
            GROQ_CHAT_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_VISION_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are Booster AI's image understanding assistant. "
                            "Describe only what is visible. Read visible text "
                            "when possible, explain diagrams and screenshots, "
                            "and clearly state uncertainty instead of guessing."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": str(question)[:4000]},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_url}
                            }
                        ]
                    }
                ],
                "temperature": 0.3,
                "max_completion_tokens": 1500
            },
            timeout=GROQ_TIMEOUT
        )

        try:
            result = response.json()
        except ValueError:
            return jsonify({
                "error": f"Invalid response from image AI (HTTP {response.status_code})."
            }), 502

        if response.status_code != 200:
            error_data = result.get("error", {}) if isinstance(result, dict) else {}
            message = (
                error_data.get("message", "Image analysis request failed.")
                if isinstance(error_data, dict)
                else "Image analysis request failed."
            )
            print(f"Groq vision error {response.status_code}: {message}")
            return jsonify({"error": message}), 502

        choices = result.get("choices", [])
        if not choices:
            return jsonify({"error": "Image AI returned no response."}), 502

        answer = choices[0].get("message", {}).get("content", "")
        if not isinstance(answer, str) or not answer.strip():
            return jsonify({"error": "Image AI returned an empty response."}), 502

        return jsonify({
            "reply": answer,
            "analysis": answer,
            "model": GROQ_VISION_MODEL
        }), 200

    except requests.exceptions.Timeout:
        return jsonify({
            "error": "Image analysis timed out. Please try again."
        }), 504
    except requests.exceptions.RequestException as exc:
        print(f"Image analysis network error: {exc}")
        return jsonify({
            "error": "Could not connect to the image AI service."
        }), 502
    except Exception as exc:
        print(f"Unexpected image analysis error: {exc}")
        return jsonify({"error": "An unexpected image analysis error occurred."}), 500


# ============================================================
# WEBSITE / WEB PROGRAM GENERATOR
# POST /generate-web
# JSON: {"prompt": "Build a responsive portfolio website..."}
# Returns generated HTML, CSS, and JavaScript in one code string.
# ============================================================
@app.route("/generate-web", methods=["POST"])
@app.route("/create-website", methods=["POST"])
def generate_web():
    try:
        if not GROQ_API_KEY:
            return jsonify({
                "error": "GROQ_API_KEY is not configured on the server."
            }), 500

        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        user_prompt = data.get("prompt") or data.get("message") or ""
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            return jsonify({
                "error": "Describe the website or web program you want to create."
            }), 400

        user_prompt = user_prompt.strip()
        if len(user_prompt) > 12000:
            user_prompt = user_prompt[:12000]

        system_prompt = """
You are Booster AI Web Developer, an expert full-stack web programmer.
Create a complete, working web project based on the user's requirements.

Rules:
- Prefer plain HTML, CSS, and JavaScript unless the user requests a framework.
- For a single-file project, return one complete HTML document with CSS in
  <style> and JavaScript in <script>.
- For a multi-file project, clearly label every file using headings such as
  FILE: index.html, FILE: style.css, and FILE: script.js.
- Include all required code; do not use placeholders such as
  "add the rest here", "same as above", or omitted sections.
- Make the interface responsive and accessible.
- Make buttons and forms perform their stated actions.
- Do not claim that code was tested or deployed unless it actually was.
- Explain any required setup, dependencies, environment variables, and run steps.
- Never expose API keys or put secrets in frontend JavaScript.
- If the request needs a backend or database, provide the required backend files
  and explain how to connect them.
Return the code and concise setup instructions.
""".strip()

        response = requests.post(
            GROQ_CHAT_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.4,
                "max_completion_tokens": 6000
            },
            timeout=GROQ_TIMEOUT
        )

        try:
            result = response.json()
        except ValueError:
            return jsonify({
                "error": f"Invalid response from web generator (HTTP {response.status_code})."
            }), 502

        if response.status_code != 200:
            error_data = result.get("error", {}) if isinstance(result, dict) else {}
            message = (
                error_data.get("message", "Website generation failed.")
                if isinstance(error_data, dict)
                else "Website generation failed."
            )
            print(f"Groq web-generation error {response.status_code}: {message}")
            return jsonify({"error": message}), 502

        choices = result.get("choices", [])
        if not choices:
            return jsonify({"error": "Web generator returned no code."}), 502

        generated_code = choices[0].get("message", {}).get("content", "")
        if not isinstance(generated_code, str) or not generated_code.strip():
            return jsonify({"error": "Web generator returned empty code."}), 502

        return jsonify({
            "reply": generated_code,
            "code": generated_code,
            "model": GROQ_MODEL,
            "type": "web_program"
        }), 200

    except requests.exceptions.Timeout:
        return jsonify({
            "error": "Website generation timed out. Please try again."
        }), 504
    except requests.exceptions.RequestException as exc:
        print(f"Web-generation network error: {exc}")
        return jsonify({
            "error": "Could not connect to the web-program generator."
        }), 502
    except Exception as exc:
        print(f"Unexpected web-generation error: {exc}")
        return jsonify({"error": "An unexpected web-generation error occurred."}), 500


# ============================================================
# CLEAR CONVERSATION HISTORY
# ============================================================
@app.route("/clear-history", methods=["POST"])
def clear_history():
    try:
        data = request.get_json(silent=True) or {}
        session_id = str(data.get("session_id") or "default")[:128]

        if session_id in conversation_history:
            conversation_history[session_id] = []
            return jsonify({"status": "History cleared"}), 200

        return jsonify({"status": "No history found"}), 200

    except Exception as exc:
        print(f"Clear-history error: {exc}")
        return jsonify({"error": "Could not clear conversation history."}), 500


# ============================================================
# HEALTH CHECK
# ============================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "Booster AI running",
        "model": GROQ_MODEL,
        "provider": "Groq",
        "vision_model": GROQ_VISION_MODEL,
        "api_key_configured": bool(GROQ_API_KEY)
    }), 200


# ============================================================
# HOME
# ============================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to Booster AI!",
        "project": "Booster AI",
        "status": "running",
        "model": GROQ_MODEL,
        "endpoints": {
            "chat": "/chat (POST)",
            "health": "/health (GET)",
            "clear-history": "/clear-history (POST)",
            "analyze-image": "/analyze-image (POST)",
            "generate-web": "/generate-web (POST)"
        },
        "docs": 'Send POST requests to /chat with {"message": "your text"}',
        "brand": "Alpha Booster | Rwanda Creative IMG"
    }), 200


# ============================================================
# ERROR HANDLERS
# ============================================================
@app.errorhandler(413)
def request_entity_too_large(_error):
    return jsonify({
        "error": "The uploaded request is too large. Please reduce file size."
    }), 413


# ============================================================
# LOCAL DEVELOPMENT
# Render runs this app using Gunicorn, not app.run().
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print("=" * 50)
    print("BOOSTER AI")
    print("=" * 50)
    print("Groq API key configured:", bool(GROQ_API_KEY))
    print("Model:", GROQ_MODEL)
    print("Conversation memory enabled (in-memory)")
    print("PDF/DOCX/TXT/code file reading enabled")
    print(f"Server: http://0.0.0.0:{port}")
    print("=" * 50)

    app.run(host="0.0.0.0", port=port, debug=False)
