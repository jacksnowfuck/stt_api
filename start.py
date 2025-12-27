import os
import sys
import threading
import torch
import gc
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from funasr import AutoModel
from datetime import datetime, timedelta
from pyannote.audio import Pipeline

# ================= 核心配置 =================
os.environ["HF_HUB_OFFLINE"] = "1"
app = Flask(__name__)

# --- 全局显存锁 (必须！) ---
# 6G 显存是极限操作，绝对不能允许多个请求撞车
gpu_lock = threading.Lock()

# --- 配置区 ---
UPLOAD_FOLDER = 'uploads'
HOTWORDS_FILE = 'hotwords.txt'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- 显存管理工具 ---
def clear_gpu():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

def move_funasr(model_obj, target_device):
    """
    负责快速搬运 FunASR (耗时约 0.5s - 1s)
    """
    if model_obj is None: return
    device = torch.device(target_device)
    try:
        if hasattr(model_obj, 'model') and model_obj.model is not None:
            model_obj.model.to(device)
        if hasattr(model_obj, 'vad_model'):
            if hasattr(model_obj.vad_model, 'model'):
                model_obj.vad_model.model.to(device)
            elif isinstance(model_obj.vad_model, torch.nn.Module):
                model_obj.vad_model.to(device)
        if hasattr(model_obj, 'punc_model'):
            if hasattr(model_obj.punc_model, 'model'):
                model_obj.punc_model.model.to(device)
            elif isinstance(model_obj.punc_model, torch.nn.Module):
                model_obj.punc_model.to(device)
    except Exception as e:
        print(f"Move FunASR Error: {e}")

# ================= 模型加载区 =================

# 1. FunASR：加载到 CPU (随时待命准备冲进 GPU)
print("1. Loading FunASR (Standby in RAM)...")
try:
    asr_model = AutoModel(
        model="paraformer-zh", model_revision="v2.0.4",
        vad_model="fsmn-vad", vad_model_revision="v2.0.4",
        punc_model="ct-punc-c", punc_model_revision="v2.0.4",
        local_files_only=False,
        device="cpu" # 初始必须是 CPU
    )
except Exception as e:
    print(f"Error loading FunASR: {e}")
    asr_model = None

# 2. Pyannote：直接焊死在 GPU 上 (避免 10s 加载延迟)
print("2. Loading Pyannote (Resident in GPU)...")
try:
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    if torch.cuda.is_available():
        print("   >>> Moving Pyannote to GPU (Occupying ~1.5GB VRAM)...")
        diarization_pipeline.to(torch.device("cuda"))
    else:
        print("   >>> No GPU found, using CPU.")
except Exception as e:
    print(f"Error loading Pyannote: {e}")
    diarization_pipeline = None


# ================= 业务逻辑 =================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_hotwords(file_path):
    if not os.path.exists(file_path): return ""
    with open(file_path, 'r', encoding='utf-8') as file:
        return ' '.join(word.strip() for word in file.readlines())

def match_speaker(text_start, text_end, diarization_segments):
    if not diarization_segments: return "Unknown"
    max_overlap = 0
    best_speaker = "Unknown"
    for seg in diarization_segments:
        overlap_start = max(text_start, seg['start'])
        overlap_end = min(text_end, seg['end'])
        overlap = max(0, overlap_end - overlap_start)
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = seg['speaker']
    return best_speaker

def clean_old_files():
    try:
        cutoff = datetime.now() - timedelta(hours=1)
        for root, _, files in os.walk(UPLOAD_FOLDER):
            for file in files:
                file_path = os.path.join(root, file)
                if datetime.fromtimestamp(os.path.getmtime(file_path)) < cutoff:
                    os.remove(file_path)
    except Exception: pass

@app.route('/api', methods=['POST'])
def api():
    if 'audio' not in request.files: return jsonify({"code": 1, "msg": "No audio"}), 400
    file = request.files['audio']
    if file.filename == '' or not allowed_file(file.filename): return jsonify({"code": 1, "msg": "Invalid file"}), 400

    filename = secure_filename(file.filename)
    date_folder = os.path.join(UPLOAD_FOLDER, datetime.now().strftime('%Y-%m-%d'))
    if not os.path.exists(date_folder): os.makedirs(date_folder)
    file_path = os.path.join(date_folder, filename)
    
    try:
        file.save(file_path)
        print(f"--- Processing: {filename} ---")

        # 【核心】抢锁：同时只允许一个人操作显卡
        with gpu_lock:
            
            # 1. 预处理 (CPU完成，不占显存)
            # 强制转 16k，能大幅降低显存消耗！这对 6G 显存至关重要
            process_file_path = os.path.join(date_folder, f"proc_{filename}.wav")
            cmd = f'ffmpeg -y -i "{file_path}" -ar 16000 -ac 1 -af "apad=pad_dur=5" -c:a pcm_s16le "{process_file_path}" -loglevel error'
            if os.system(cmd) != 0: raise Exception("FFmpeg failed")
            abs_path = os.path.abspath(process_file_path)

            # 2. 运行 Pyannote (它已经在 GPU 里了，直接跑)
            diarization_segments = []
            if diarization_pipeline:
                print(">> [1/3] Pyannote Diarization (GPU)...")
                try:
                    # 此时显存 = Pyannote 静态 + Pyannote 动态 (约 4G)
                    # FunASR 在 CPU，不干扰
                    diarization = diarization_pipeline(abs_path)
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        diarization_segments.append({
                            "start": turn.start, "end": turn.end, "speaker": speaker
                        })
                except Exception as e:
                    print(f"Pyannote Error: {e}")
                
                # 跑完赶紧清理动态显存，变回静态待机状态
                clear_gpu()

            # 3. 运行 FunASR (动态搬进去，用完踢出来)
            final_subtitles = []
            if asr_model:
                print(">> [2/3] Moving FunASR to GPU...")
                move_funasr(asr_model, "cuda") # 耗时约 0.5s
                
                print(">> [3/3] FunASR Transcribing (GPU)...")
                try:
                    # 此时显存 = Pyannote 静态 (1.5G) + FunASR 动态 (3G) = 4.5G < 6G
                    # 安全！
                    res = asr_model.generate(
                        input=process_file_path,
                        return_raw_text=True,
                        is_final=True,
                        sentence_timestamp=True,
                        batch_size_s=60,
                        hotword=load_hotwords(HOTWORDS_FILE),
                        device="cuda"
                    )
                    
                    if res and 'sentence_info' in res[0]:
                        for i, item in enumerate(res[0]['sentence_info']):
                            start_s = item['timestamp'][0][0] / 1000.0
                            end_s = item['timestamp'][-1][1] / 1000.0
                            speaker = match_speaker(start_s, end_s, diarization_segments)
                            final_subtitles.append({
                                "line": i + 1, "speaker": speaker,
                                "start_time": start_s, "end_time": end_s, "text": item['text']
                            })
                except Exception as e:
                    print(f"FunASR Error: {e}")
                
                finally:
                    # 【关键】必须踢走 FunASR，否则下次 Pyannote 跑的时候会爆
                    print(">> Offloading FunASR to RAM...")
                    move_funasr(asr_model, "cpu")
                    clear_gpu()

        # ================= 流程结束 =================

        if os.path.exists(process_file_path): os.remove(process_file_path)
        clean_old_files()

        print("\n" + "="*30)
        for item in final_subtitles:
            print(f"{item['speaker']}: {item['text']}")
        print("="*30 + "\n")

        return jsonify({"code": 0, "msg": 'ok', "data": final_subtitles})

    except Exception as e:
        print(f"Critical Error: {e}")
        # 出错兜底：确保 FunASR 被踢回 CPU
        if asr_model: move_funasr(asr_model, "cpu")
        clear_gpu()
        return jsonify({"code": 2, "msg": str(e)}), 500

if __name__ == '__main__':
    print("Server starting at http://0.0.0.0:9933")
    app.run(debug=False, host="0.0.0.0", port=9933, threaded=True)
