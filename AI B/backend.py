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
