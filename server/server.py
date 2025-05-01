# server.py
import os
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

from .agent import get_response             
from tools.entity_recognition.ingredient_recognition import ingredients_detector
from tools.audio.speech_to_text import transcribe_audio
from tools.grocery_search.grocery_helper import search_grocery_store_nearby
import tempfile, subprocess, mimetypes

# SINGLE instance – point to front_end
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
FRONT_DIR = os.path.join(BASE_DIR, "front_end")
app = Flask(__name__, static_folder=FRONT_DIR, static_url_path="")
# CONFIGURATION
UPLOAD_DIR = "uploads"
# purge old files on start, create fresh dir
import shutil, uuid
if os.path.exists(UPLOAD_DIR):
    shutil.rmtree(UPLOAD_DIR)
os.makedirs(UPLOAD_DIR)

# Load credentials from env
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SPEECH_KEY     = os.environ["SPEECH_KEY"]
SPEECH_REGION  = os.environ["SPEECH_REGION"]

if not SPEECH_KEY or not SPEECH_REGION:
    raise RuntimeError("SPEECH_KEY / SPEECH_REGION not set in environment")

CORS(app)  # allow index.html (file:// or other port) to talk to this server

def _session_id() -> str:
    return request.cookies.get("sid") or request.remote_addr or "anon"

# Serve your index.html at the root
@app.route("/")
def index():
    return app.send_static_file("index.html")

# 1. Text endpoint
@app.route("/api/text", methods=["POST"])
def api_text():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # call into your agent
    resp = get_response(text, _session_id())
    return jsonify({"response": resp})

# 2. Image endpoint
@app.route("/api/image", methods=["POST"])
def api_image():
    img = request.files.get("image")
    if not img:
        return jsonify({"error": "No image uploaded"}), 400

    ext = os.path.splitext(img.filename)[1]
    filename = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, filename)
    img.save(path)

    # first do entity_recognition → text
    img_text = ingredients_detector(path)

    # then send that to agent
    resp = get_response(img_text, _session_id())
    return jsonify({"response": resp})

# 3. Speech endpoint
@app.route("/api/speech", methods=["POST"])
def api_speech():
    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "No audio uploaded"}), 400
    
        # save to tmp file
    ext = mimetypes.guess_extension(audio.mimetype) or ".webm"
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    audio.save(tmp_in.name)

    # if WebM/OGG → convert to wav 16 kHz for Azure
    if audio.mimetype != "audio/wav":
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", tmp_in.name,
                    "-acodec", "pcm_s16le",  # 16-bit PCM
                    "-ar", "16000",          # 16 kHz
                    "-ac", "1",              # mono
                    tmp_wav
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            app.logger.error("ffmpeg not found — install it or upload wav/16k mono")
            return jsonify({"error": "ffmpeg missing on server"}), 500
        except subprocess.CalledProcessError as e:
            app.logger.exception("ffmpeg failed: %s", e)
            return jsonify({"error": "audio conversion failed"}), 500
    else:
        tmp_wav = tmp_in.name

    try:
        text = transcribe_audio(tmp_wav, speech_key=SPEECH_KEY, region=SPEECH_REGION)
    except Exception as e:
        app.logger.exception("STT error")
        return jsonify({"error": str(e)}), 500

    if not text:
        return jsonify({"error": "No speech recognized, please try again."}), 200

    resp = get_response(text, _session_id())
    return jsonify({"response": resp, "transcript": text})

# 4. Grocery search  (frontend calls this after it gets zip code from user)
@app.route("/api/grocery", methods=["POST"])
def api_grocery():
    data = request.get_json()
    zipcode = data.get("zip", "").strip()
    items   = data.get("items", [])
    if not zipcode or not items:
        return jsonify({"error": "zip or items missing"}), 400
    try:
        stores = search_grocery_store_nearby(zipcode, items, radius=3500)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"stores": stores})

def stream_openai(answer_gen):
    def gen():
        for chunk in answer_gen:
            yield f"data:{chunk}\n\n"
    return Response(gen(), mimetype="text/event-stream")

if __name__ == "__main__":
    # debug=True only for local dev
    app.run(host="0.0.0.0", port=5000, debug=True)
