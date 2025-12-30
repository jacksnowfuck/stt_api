import os
import sys
import threading
import torch
import gc
import traceback
import subprocess  # 用于安全的命令行执行
import time        # 用于 I/O 等待
import wave        # [新增] 用于读取音频时长
import math        # [新增] 用于向上取整计算
import contextlib  # [新增] 用于安全打开文件
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from funasr import AutoModel
from datetime import datetime, timedelta
from pyannote.audio import Pipeline

# ================= 核心配置 =================
os.environ["HF_HUB_OFFLINE"] = "1"
# 强制 PyTorch 使用确定性算法
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

app = Flask(__name__)

# --- 全局显存锁 ---
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
        torch.cuda.synchronize() # 强制同步

def move_funasr(model_obj, target_device):
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
        
        # 搬运后强制同步
        if target_device == "cuda":
            torch.cuda.synchronize()
            
    except Exception as e:
        print(f"Move FunASR Error: {e}")

# ================= 模型加载区 =================
print("1. Loading FunASR (Standby in RAM)...")
try:
    asr_model = AutoModel(
        model="paraformer-zh", model_revision="v2.0.4",
        vad_model="fsmn-vad", vad_model_revision="v2.0.4",
        punc_model="ct-punc-c", punc_model_revision="v2.0.4",
        local_files_only=False,
        device="cpu"
    )
except Exception as e:
    print(f"Error loading FunASR: {e}")
    asr_model = None

print("2. Loading Pyannote (Resident in GPU)...")
try:
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    if torch.cuda.is_available():
        print("   >>> Moving Pyannote to GPU...")
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

def match_speaker(text_start, text_end, diarization_segments, last_speaker="Unknown"):
    if not diarization_segments: return "Unknown"
    max_overlap = 0
    best_speaker = None
    for seg in diarization_segments:
        overlap_start = max(text_start, seg['start'])
        overlap_end = min(text_end, seg['end'])
        overlap = max(0, overlap_end - overlap_start)
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = seg['speaker']
    if best_speaker:
        return best_speaker
    last_segment_end = diarization_segments[-1]['end']
    if text_start >= last_segment_end:
        return last_speaker if last_speaker != "Unknown" else diarization_segments[-1]['speaker']
    return "Unknown"

def clean_old_files():
    try:
        cutoff = datetime.now() - timedelta(hours=1)
        for root, _, files in os.walk(UPLOAD_FOLDER):
            for file in files:
                file_path = os.path.join(root, file)
                if datetime.fromtimestamp(os.path.getmtime(file_path)) < cutoff:
                    os.remove(file_path)
    except Exception: pass

# [新增] 获取音频时长的辅助函数
def get_wav_duration(file_path):
    with contextlib.closing(wave.open(file_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

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

        with gpu_lock:
            # ================= 1. 预处理 (FFmpeg) =================
            process_file_path = os.path.join(date_folder, f"proc_{filename}.wav")
            abs_path = os.path.abspath(process_file_path)
            
            # Padding 保持 5秒
            PADDING_DURATION = 5.0 
            cmd = [
                'ffmpeg', '-y', '-i', file_path,
                '-ar', '16000', '-ac', '1',
                f'-af', f'apad=pad_dur={PADDING_DURATION}', # 填充5秒静音
                '-c:a', 'pcm_s16le',
                abs_path
            ]
            
            print(">> [0/3] Preprocessing Audio (FFmpeg)...")
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(0.5)  # 强制等待IO
                if not os.path.exists(abs_path):
                    raise Exception("FFmpeg output file not found")
            except subprocess.CalledProcessError as e:
                raise Exception(f"FFmpeg failed with code {e.returncode}")

            # [关键] 获取处理后的总时长（包含Padding）
            total_wav_duration = get_wav_duration(abs_path)

            # ================= 2. Pyannote =================
            diarization_segments = []
            if diarization_pipeline:
                print(">> [1/3] Pyannote Diarization (GPU)...")
                try:
                    clear_gpu()
                    diarization = diarization_pipeline(abs_path)
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        diarization_segments.append({
                            "start": turn.start, "end": turn.end, "speaker": speaker
                        })
                except Exception as e:
                    print(f"Pyannote Error: {e}")
                clear_gpu()

            # ================= 3. FunASR =================
            final_subtitles = []
            speaker_map = {} 
            next_speaker_idx = 0
            def get_display_speaker(raw_name):
                nonlocal next_speaker_idx
                if raw_name not in speaker_map:
                    speaker_map[raw_name] = str(next_speaker_idx)
                    next_speaker_idx += 1
                return speaker_map[raw_name]

            if asr_model:
                print(">> [2/3] Moving FunASR to GPU...")
                move_funasr(asr_model, "cuda")
                
                # GPU 预热与同步
                torch.cuda.synchronize()
                
                # ========= 【核心修复：计算动态整数 Batch Size】 =========
                MAX_BATCH_LIMIT = 50.0 
                
                if total_wav_duration <= MAX_BATCH_LIMIT:
                    # 如果短于50秒，直接向上取整变整数
                    dynamic_batch_size = int(math.ceil(total_wav_duration))
                else:
                    # 如果长，先算出分几份 (向上取整)
                    num_batches = math.ceil(total_wav_duration / MAX_BATCH_LIMIT)
                    # 再算出每一份的大小 (向上取整)
                    # 例如: 91秒 / 2 = 45.5 -> ceil -> 46 (Int)
                    dynamic_batch_size = int(math.ceil(total_wav_duration / num_batches))
                
                print(f">> [3/3] FunASR Transcribing (GPU)...")
                print(f"   >>> [Dynamic Batching] Total: {total_wav_duration:.2f}s | Batch Size: {dynamic_batch_size} (Integer)")
                # ========================================================

                try:
                    res = asr_model.generate(
                        input=process_file_path,
                        return_raw_text=True,
                        is_final=True,
                        sentence_timestamp=True,
                        
                        # 【修复点】应用计算出的动态整数
                        batch_size_s=dynamic_batch_size, 
                        
                        hotword=load_hotwords(HOTWORDS_FILE),
                        device="cuda",
                        # 显式指定参数，防止模型内部默认值的非确定性行为
                        vad_kwargs={
                            "max_single_segment_time": 60000,
                            "vad_tail_silence_time": 800, # 尾部静音阈值
                        }
                    )
                    
                    full_text_str = ""
                    last_valid_raw_speaker = "Unknown" 
                    last_end_time = 0.0

                    if res and 'sentence_info' in res[0]:
                        for i, item in enumerate(res[0]['sentence_info']):
                            start_s = item['timestamp'][0][0] / 1000.0
                            end_s = item['timestamp'][-1][1] / 1000.0
                            raw_speaker = match_speaker(start_s, end_s, diarization_segments, last_valid_raw_speaker)
                            display_speaker = get_display_speaker(raw_speaker)
                            last_valid_raw_speaker = raw_speaker
                            last_end_time = end_s
                            
                            final_subtitles.append({
                                "line": len(final_subtitles) + 1, 
                                "speaker": display_speaker, 
                                "start_time": start_s, 
                                "end_time": end_s, 
                                "text": item['text']
                            })
                            full_text_str += item['text']

                    # --- 防吞字补丁 (兜底策略) ---
                    if res and 'text' in res[0]:
                        raw_text = res[0]['text'].replace(" ", "")
                        processed_text = full_text_str.replace(" ", "")
                        
                        if len(raw_text) > len(processed_text) + 2:
                            print(f">>> PATCHING! Gap: {len(raw_text) - len(processed_text)}")
                            missing_part = res[0]['text'][len(full_text_str):]
                            if missing_part.strip():
                                final_subtitles.append({
                                    "line": len(final_subtitles) + 1,
                                    "speaker": get_display_speaker(last_valid_raw_speaker),
                                    "start_time": last_end_time,
                                    "end_time": last_end_time + 2.0,
                                    "text": missing_part.strip()
                                })
                    # ----------------------------

                except Exception as e:
                    print(f"FunASR Error: {e}")
                    traceback.print_exc()
                finally:
                    print(">> Offloading FunASR to RAM...")
                    move_funasr(asr_model, "cpu")
                    clear_gpu()

        if os.path.exists(process_file_path): os.remove(process_file_path)
        clean_old_files()

        print("\n" + "="*30)
        for item in final_subtitles:
            print(f"{item['speaker']}: {item['text']}")
        print("="*30 + "\n")

        return jsonify({"code": 0, "msg": 'ok', "data": final_subtitles})

    except Exception as e:
        print(f"Critical Error: {e}")
        if asr_model: move_funasr(asr_model, "cpu")
        clear_gpu()
        return jsonify({"code": 2, "msg": str(e)}), 500

if __name__ == '__main__':
    print("Server starting at http://0.0.0.0:9933")
    app.run(debug=False, host="0.0.0.0", port=9933, threaded=True)
