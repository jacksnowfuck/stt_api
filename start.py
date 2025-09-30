from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import gc
import torch
from funasr import AutoModel
from datetime import datetime, timedelta
import glob
import soundfile as sf

def audio_file_is_valid(file_path):
    try:
        # 能够打开并读取代表文件是有效音频
        with sf.SoundFile(file_path) as f:
            return f.frames > 0
    except Exception:
        return False

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'

# Load hotwords
def load_hotwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        hotwords = file.readlines()
    return ' '.join(word.strip() for word in hotwords)

# Initialize model
def load_model():
    model = AutoModel(
        model="paraformer-zh", model_revision="v2.0.4",
        vad_model="fsmn-vad", vad_model_revision="v2.0.4",
        punc_model="ct-punc-c", punc_model_revision="v2.0.4",
        local_files_only=False
    )
    return model

model = load_model()
hotwords_str = load_hotwords('hotwords.txt')
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_and_release_memory():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        available_memory = total_memory - reserved_memory
        
        available_memory_mb = available_memory / 1024 / 1024
        print(f"Available GPU Memory: {available_memory_mb:.2f} MB")
        
        if available_memory_mb < 50:
            print("Low GPU memory detected. Clearing cache.")
            torch.cuda.empty_cache()

def clean_old_files():
    # Get the current time
    now = datetime.now()
    # Calculate one hour ago
    cutoff = now - timedelta(hours=1)
    # Iterate over all files in the uploads dir
    for root, _, files in os.walk(UPLOAD_FOLDER):
        for file in files:
            file_path = os.path.join(root, file)
            # Get the last modified time and compare
            if datetime.fromtimestamp(os.path.getmtime(file_path)) < cutoff:
                print(f"Deleting old file: {file_path}")
                os.remove(file_path)

@app.route('/api', methods=['POST'])
def api():
    if 'audio' not in request.files:
        return jsonify({"code": 1, "msg": "No audio file uploaded"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"code": 1, "msg": "No audio file uploaded"}), 400

    if file and allowed_file(file.filename):
        # Create a directory for today's date
        date_folder = os.path.join(UPLOAD_FOLDER, datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(date_folder, exist_ok=True)
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(date_folder, filename)
        file.save(file_path)

        try:
            print("Starting audio processing")

            # Debug: Verify that the audio file is valid
            if not audio_file_is_valid(file_path):
                print(f"Uploaded file is empty or invalid audio")
                return jsonify({"code": 1, "msg": "Uploaded file is empty or invalid audio"}), 400

            # Set a smaller batch size to reduce memory usage
            res = model.generate(
                input=file_path,
                return_raw_text=True,
                is_final=True,
                sentence_timestamp=True,
                batch_size_s=100,
                hotword=hotwords_str
            )

            raw_subtitles = []
            for it in res[0]['sentence_info']:
                raw_subtitles.append({
                    "line": len(raw_subtitles) + 1,
                    "speaker": '0',
                    "text": it['text']
                })

            return jsonify({"code": 0, "msg": 'ok', "data": raw_subtitles})
        except Exception as e:
            print(f"Exception during audio processing: {e}")
            return jsonify({"code": 2, "msg": str(e)}), 500
        finally:
            check_and_release_memory()
            gc.collect()
            clean_old_files()
    else:
        return jsonify({"code": 1, "msg": f"Unsupported file format {file.filename.rsplit('.', 1)[1].lower()}"}), 400

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port="9933")
