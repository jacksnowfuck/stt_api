import os
import sys
import threading
import torch
import gc
import json
import traceback
import subprocess
import time
import wave
import math
import contextlib
import difflib
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from funasr import AutoModel
from datetime import datetime, timedelta
from pyannote.audio import Pipeline

# ================= 核心配置 =================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MODELSCOPE_OFFLINE"] = "1"

# ==========================================
# [日志配置]
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 日志文件名保持不变，用于记录缺失警告
LOG_FILE_PATH = os.path.join(CURRENT_DIR, 'asr_missing.log') 

root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING) # 全局只看 WARNING 以上，屏蔽 INFO

# 特权：让 Flask 的请求日志正常显示
logging.getLogger('werkzeug').setLevel(logging.INFO)

if root_logger.hasHandlers():
    root_logger.handlers.clear()

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 1. 文件 Handler (记录缺失警告)
file_handler = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# 2. 控制台 Handler (显示警告和 Flask 日志)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

print(f"✅ Logging Configured. Missing warnings will be saved to: {LOG_FILE_PATH}")
# ==========================================

app = Flask(__name__)

# --- 全局显存锁 ---
gpu_lock = threading.Lock()

# --- 基础配置 ---
UPLOAD_FOLDER = 'uploads'
HOTWORDS_FILE = 'hotwords.txt'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg'}

# --- 噪音门配置 ---
NOISE_GATE_FILTER = "afftdn=nr=20,compand=attacks=0:decays=0.01:points=-90/-90|-30/-90|-25/-25|0/0:gain=0"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ================= 工具函数 =================

def clear_gpu():
    """彻底清理显存"""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def move_funasr(model_obj, target_device):
    """动态搬运 FunASR 模型"""
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
        
        if target_device == "cuda":
            torch.cuda.synchronize()
            
    except Exception as e:
        print(f"Move FunASR Error: {e}")

# ================= 模型加载 =================
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

# ================= 业务辅助逻辑 =================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_hotwords(file_path):
    if not os.path.exists(file_path): return ""
    with open(file_path, 'r', encoding='utf-8') as file:
        return ' '.join(word.strip() for word in file.readlines())

def get_audio_channels(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.wav':
        try:
            with wave.open(file_path, 'rb') as wav_file:
                return wav_file.getnchannels()
        except: pass
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=channels', '-of', 'json', file_path]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        info = json.loads(result.stdout)
        if 'streams' in info and len(info['streams']) > 0:
            return int(info['streams'][0]['channels'])
        return 1
    except: return 1

def calculate_dynamic_batch_size(duration):
    MAX_BATCH_LIMIT = 50.0
    if duration <= MAX_BATCH_LIMIT:
        return int(math.ceil(duration))
    else:
        num_batches = math.ceil(duration / MAX_BATCH_LIMIT)
        return int(math.ceil(duration / num_batches))

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
    if best_speaker: return best_speaker
    last_segment_end = diarization_segments[-1]['end']
    if text_start >= last_segment_end:
        return last_speaker if last_speaker != "Unknown" else diarization_segments[-1]['speaker']
    return "Unknown"

def get_wav_duration(file_path):
    try:
        with contextlib.closing(wave.open(file_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)
    except: return 0.0

def remove_echo_segments(segments, similarity_threshold=0.6, overlap_threshold=0.3):
    if not segments: return []
    segments.sort(key=lambda x: x['start_time'])
    indices_to_remove = set()
    n = len(segments)
    for i in range(n):
        if i in indices_to_remove: continue
        seg_a = segments[i]
        for j in range(i + 1, n):
            seg_b = segments[j]
            if seg_b['start_time'] - seg_a['end_time'] > 3.0: break
            if seg_a['speaker'] == seg_b['speaker']: continue
            start_max = max(seg_a['start_time'], seg_b['start_time'])
            end_min = min(seg_a['end_time'], seg_b['end_time'])
            overlap = max(0, end_min - start_max)
            min_dur = min(seg_a['end_time']-seg_a['start_time'], seg_b['end_time']-seg_b['start_time'])
            if min_dur <= 0: continue
            if (overlap / min_dur) > overlap_threshold:
                ratio = difflib.SequenceMatcher(None, seg_a['text'], seg_b['text']).ratio()
                if ratio > similarity_threshold:
                    if seg_a['speaker'] == 0 and seg_b['speaker'] == 1: indices_to_remove.add(i)
                    elif seg_b['speaker'] == 0 and seg_a['speaker'] == 1: indices_to_remove.add(j)
                    else:
                        if seg_a['start_time'] > seg_b['start_time']: indices_to_remove.add(i)
                        else: indices_to_remove.add(j)
    return [segments[i] for i in range(n) if i not in indices_to_remove]

def clean_old_files():
    try:
        cutoff = datetime.now() - timedelta(hours=1)
        for root, _, files in os.walk(UPLOAD_FOLDER):
            for file in files:
                file_path = os.path.join(root, file)
                if datetime.fromtimestamp(os.path.getmtime(file_path)) < cutoff:
                    try: os.remove(file_path)
                    except: pass
    except: pass

# ================= API 核心接口 =================

@app.route('/api', methods=['POST'])
def api():
    if 'audio' not in request.files:
        return jsonify({"code": 1, "msg": "No audio file provided"}), 400
    file = request.files['audio']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"code": 1, "msg": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    date_folder = os.path.join(UPLOAD_FOLDER, datetime.now().strftime('%Y-%m-%d'))
    if not os.path.exists(date_folder): os.makedirs(date_folder)
    original_path = os.path.join(date_folder, filename)
    
    temp_files_to_clean = [] 

    try:
        file.save(original_path)
        print(f"\n--- New Task: {filename} ---")
        
        final_subtitles = []
        current_audio_duration = 0.0

        with gpu_lock:
            # 移除了所有 for 循环和重试逻辑，回归单次执行
            
            channels = get_audio_channels(original_path)
            hotwords = load_hotwords(HOTWORDS_FILE)
            print(f"    >>> Detected Channels: {channels}")

            # =======================================================
            # 分支 A: 双声道立体声 (Stereo)
            # =======================================================
            if channels == 2:
                print("    >>> [Mode] Stereo Split with Strict VAD")
                base_name = os.path.splitext(original_path)[0]
                left_path = f"{base_name}_left.wav"
                right_path = f"{base_name}_right.wav"
                
                if left_path not in temp_files_to_clean:
                    temp_files_to_clean.extend([left_path, right_path])

                filter_left = f"pan=mono|c0=c0,highpass=f=100,{NOISE_GATE_FILTER},loudnorm=I=-16:TP=-1.5:LRA=11"
                filter_right = f"pan=mono|c0=c1,highpass=f=100,{NOISE_GATE_FILTER},loudnorm=I=-16:TP=-1.5:LRA=11"

                subprocess.run(['ffmpeg', '-y', '-i', original_path, '-af', filter_left, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', left_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(['ffmpeg', '-y', '-i', original_path, '-af', filter_right, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', right_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                current_audio_duration = get_wav_duration(left_path)

                if asr_model:
                    move_funasr(asr_model, "cuda") 
                    torch.cuda.synchronize()

                    tasks = [{"path": left_path, "speaker": 0}, {"path": right_path, "speaker": 1}]
                    all_segments = []

                    for task in tasks:
                        t_path = task["path"]
                        t_speaker = task["speaker"]
                        
                        duration = get_wav_duration(t_path)
                        dynamic_batch = calculate_dynamic_batch_size(duration)
                        print(f"      -> Processing Speaker {t_speaker}...")

                        res = asr_model.generate(
                            input=t_path,
                            return_raw_text=True,
                            is_final=True,
                            sentence_timestamp=True, 
                            batch_size_s=dynamic_batch, 
                            hotword=hotwords,
                            device="cuda",
                            vad_kwargs={
                                "max_single_segment_time": 6000, 
                                "vad_tail_silence_time": 400,    
                            }
                        )

                        if res and 'sentence_info' in res[0]:
                            for item in res[0]['sentence_info']:
                                text_content = item['text'].strip()
                                if not text_content: continue
                                timestamps = item.get('timestamp', [])
                                if not timestamps:
                                    all_segments.append({
                                        "speaker": t_speaker,
                                        "start_time": item['timestamp'][0][0] / 1000.0,
                                        "end_time": item['timestamp'][-1][1] / 1000.0,
                                        "text": text_content
                                    })
                                    continue
                                
                                sub_segments_indices = []
                                current_indices = []
                                for i, (ws, we) in enumerate(timestamps):
                                    if i > 0:
                                        prev_end = timestamps[i-1][1]
                                        if ws - prev_end > 800:
                                            if current_indices: sub_segments_indices.append(current_indices)
                                            current_indices = []
                                    current_indices.append(i)
                                if current_indices: sub_segments_indices.append(current_indices)

                                if len(sub_segments_indices) == 1:
                                    all_segments.append({
                                        "speaker": t_speaker,
                                        "start_time": timestamps[0][0] / 1000.0,
                                        "end_time": timestamps[-1][1] / 1000.0,
                                        "text": text_content
                                    })
                                else:
                                    cursor_text = 0
                                    total_stamps = len(timestamps)
                                    text_len = len(text_content)
                                    for idx, indices in enumerate(sub_segments_indices):
                                        seg_start = timestamps[indices[0]][0] / 1000.0
                                        seg_end = timestamps[indices[-1]][1] / 1000.0
                                        ratio = len(indices) / total_stamps
                                        char_count = int(text_len * ratio)
                                        if idx == len(sub_segments_indices) - 1:
                                            seg_text = text_content[cursor_text:]
                                        else:
                                            split_p = cursor_text + char_count
                                            while split_p < text_len and text_content[split_p] not in ['，','。','？','！',',','.','?','!',' ']:
                                                split_p += 1
                                            seg_text = text_content[cursor_text : split_p+1]
                                            cursor_text = split_p + 1
                                        if not seg_text.strip(): continue
                                        all_segments.append({
                                            "speaker": t_speaker,
                                            "start_time": seg_start,
                                            "end_time": seg_end,
                                            "text": seg_text.strip()
                                        })

                    all_segments = remove_echo_segments(all_segments)
                    all_segments.sort(key=lambda x: x['start_time'])
                    real_line_idx = 1
                    for seg in all_segments:
                        if not seg['text'].strip(): continue 
                        seg['line'] = real_line_idx
                        real_line_idx += 1
                        final_subtitles.append(seg)
                    
                    move_funasr(asr_model, "cpu")
                    clear_gpu()

            # =======================================================
            # 分支 B: 单声道/其他 (Mono)
            # =======================================================
            else:
                print("    >>> [Mode] Mono Diarization")
                process_file_path = os.path.join(date_folder, f"proc_{filename}.wav")
                abs_path = os.path.abspath(process_file_path)
                
                if process_file_path not in temp_files_to_clean:
                    temp_files_to_clean.append(process_file_path)
                
                subprocess.run(['ffmpeg', '-y', '-i', original_path, '-ar', '16000', '-ac', '1', '-af', f"{NOISE_GATE_FILTER},loudnorm=I=-16:TP=-1.5:LRA=11", abs_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                current_audio_duration = get_wav_duration(abs_path)

                diarization_segments = []
                if diarization_pipeline:
                    try:
                        clear_gpu()
                        diarization = diarization_pipeline(abs_path)
                        for turn, _, speaker in diarization.itertracks(yield_label=True):
                            diarization_segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
                    except Exception as e:
                        print(f"Pyannote Error: {e}")
                    clear_gpu()

                speaker_map = {} 
                next_speaker_idx = 0
                def get_speaker_int(raw):
                    nonlocal next_speaker_idx
                    if raw not in speaker_map:
                        speaker_map[raw] = next_speaker_idx 
                        next_speaker_idx += 1
                    return speaker_map[raw]

                if asr_model:
                    move_funasr(asr_model, "cuda")
                    torch.cuda.synchronize()
                    
                    res = asr_model.generate(
                        input=process_file_path,
                        return_raw_text=True,
                        is_final=True,
                        sentence_timestamp=True,
                        batch_size_s=calculate_dynamic_batch_size(current_audio_duration), 
                        hotword=hotwords,
                        device="cuda",
                        vad_kwargs={
                            "max_single_segment_time": 6000, 
                            "vad_tail_silence_time": 400,
                        }
                    )
                    
                    full_text_str = ""
                    last_valid_spk = "Unknown"
                    last_end = 0.0

                    if res and 'sentence_info' in res[0]:
                        for item in res[0]['sentence_info']:
                            text_content = item['text'].strip()
                            if not text_content: continue 

                            start_s = item['timestamp'][0][0] / 1000.0
                            end_s = item['timestamp'][-1][1] / 1000.0
                            
                            raw_spk = match_speaker(start_s, end_s, diarization_segments, last_valid_spk)
                            final_subtitles.append({
                                "line": len(final_subtitles) + 1, 
                                "speaker": get_speaker_int(raw_spk), 
                                "start_time": start_s, 
                                "end_time": end_s, 
                                "text": text_content
                            })
                            
                            last_valid_spk = raw_spk
                            last_end = end_s
                            full_text_str += item['text']
                            
                        if 'text' in res[0]:
                            raw_t = res[0]['text'].replace(" ", "")
                            proc_t = full_text_str.replace(" ", "")
                            if len(raw_t) > len(proc_t) + 2:
                                    missing = res[0]['text'][len(full_text_str):]
                                    if missing.strip(): 
                                        final_subtitles.append({
                                            "line": len(final_subtitles) + 1,
                                            "speaker": get_speaker_int(last_valid_spk),
                                            "start_time": last_end,
                                            "end_time": last_end + 2.0,
                                            "text": missing.strip()
                                        })

                    move_funasr(asr_model, "cpu")
                    clear_gpu()

            # =======================================================
            # 逻辑：检查完整性并记录日志（但不重试）
            # =======================================================
            last_subtitle_end = 0.0
            if final_subtitles:
                last_subtitle_end = final_subtitles[-1]['end_time']
            
            time_diff = current_audio_duration - last_subtitle_end
            
            print(f"    >>> Check: File={filename}, Total={current_audio_duration:.2f}s, LastSub={last_subtitle_end:.2f}s, Diff={time_diff:.2f}s")
            
            # 如果误差大于5秒，记录 Warning 日志到文件
            if time_diff > 5.0:
                warn_msg = f"File {filename}: INCOMPLETE. Missing {time_diff:.2f}s (Total: {current_audio_duration:.2f}s, End: {last_subtitle_end:.2f}s)"
                print(f"    >>> [WARNING] {warn_msg}")
                logging.warning(warn_msg)
            else:
                print(f"    >>> [Success] Time check passed.")

        for tmp in temp_files_to_clean:
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except: pass
        clean_old_files()

        print("\n" + "="*50)
        print(f"Result for: {filename} (Total: {len(final_subtitles)})")
        print("-" * 50)
        for item in final_subtitles:
            print(f"[{item['start_time']:>7.2f}] Speaker {item['speaker']}: {item['text']}")
        print("="*50 + "\n")

        return jsonify({"code": 0, "msg": 'ok', "data": final_subtitles})

    except Exception as e:
        err_msg = f"CRITICAL ERROR for {filename}: {e}"
        print(err_msg)
        traceback.print_exc()
        logging.error(err_msg) # 异常也要记录
        if asr_model: move_funasr(asr_model, "cpu")
        clear_gpu()
        return jsonify({"code": 2, "msg": str(e)}), 500

if __name__ == '__main__':
    print("Server starting at http://0.0.0.0:9933")
    app.run(debug=False, host="0.0.0.0", port=9933, threaded=True)
