import os
import sys
import threading
import torch
import gc
import json
import traceback
import subprocess  # 用于安全的命令行执行
import time        # 用于 I/O 等待
import wave        # 用于读取音频时长
import math        # 用于向上取整计算
import contextlib  # 用于安全打开文件
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from funasr import AutoModel
from datetime import datetime, timedelta
from pyannote.audio import Pipeline

# ================= 核心配置 =================
# 设置离线模式，防止运行时尝试连接 HuggingFace/ModelScope
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MODELSCOPE_OFFLINE"] = "1"

# 强制 PyTorch 使用确定性算法 (尽可能保证结果一致性)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

app = Flask(__name__)

# --- 全局显存锁 ---
# 确保同一时间只有一个请求能使用 GPU，防止显存爆炸
gpu_lock = threading.Lock()

# --- 基础配置 ---
UPLOAD_FOLDER = 'uploads'
HOTWORDS_FILE = 'hotwords.txt'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ================= 工具函数 =================

def clear_gpu():
    """彻底清理显存"""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize() # 强制同步等待清理完成

def move_funasr(model_obj, target_device):
    """
    动态将 FunASR 模型在 CPU (内存) 和 GPU (显存) 之间搬运。
    用完即走，释放显存给 Pyannote 或其他任务。
    """
    if model_obj is None: return
    device = torch.device(target_device)
    try:
        # 搬运主模型
        if hasattr(model_obj, 'model') and model_obj.model is not None:
            model_obj.model.to(device)
        # 搬运 VAD 模型
        if hasattr(model_obj, 'vad_model'):
            if hasattr(model_obj.vad_model, 'model'):
                model_obj.vad_model.model.to(device)
            elif isinstance(model_obj.vad_model, torch.nn.Module):
                model_obj.vad_model.to(device)
        # 搬运标点模型
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

# ================= 模型加载 =================
print("1. Loading FunASR (Standby in RAM)...")
try:
    # 默认加载到 CPU，只有需要识别时才搬运到 GPU
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
    # Pyannote 加载较慢，通常常驻 GPU 或内存
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    if torch.cuda.is_available():
        print("   >>> Moving Pyannote to GPU...")
        diarization_pipeline.to(torch.device("cuda"))
    else:
        print("   >>> No GPU found, using CPU (Warning: Very Slow).")
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
    """使用 ffprobe 检测音频声道数"""
    cmd = [
        'ffprobe', '-v', 'error', 
        '-select_streams', 'a:0', 
        '-show_entries', 'stream=channels', 
        '-of', 'json', file_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        info = json.loads(result.stdout)
        return int(info['streams'][0]['channels'])
    except Exception as e:
        print(f"Error checking channels: {e}, assuming Mono (1).")
        return 1

def calculate_dynamic_batch_size(duration):
    """
    根据音频时长动态计算 batch_size_s。
    防止长音频导致显存溢出 (OOM)。
    """
    MAX_BATCH_LIMIT = 50.0  # 单次最大处理时长建议值
    if duration <= MAX_BATCH_LIMIT:
        return int(math.ceil(duration))
    else:
        # 如果音频很长，切分成多份，每份尽量接近 MAX_BATCH_LIMIT
        num_batches = math.ceil(duration / MAX_BATCH_LIMIT)
        return int(math.ceil(duration / num_batches))

def match_speaker(text_start, text_end, diarization_segments, last_speaker="Unknown"):
    """将 ASR 文本的时间戳与声纹分离的时间戳进行匹配"""
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
    
    # 如果找到匹配的说话人
    if best_speaker:
        return best_speaker
    
    # 如果没找到重叠，检查是否紧跟在最后一段之后
    last_segment_end = diarization_segments[-1]['end']
    if text_start >= last_segment_end:
        # 如果之前的 speaker 也是未知的，就用最后一段的 speaker
        return last_speaker if last_speaker != "Unknown" else diarization_segments[-1]['speaker']
    
    return "Unknown"

def get_wav_duration(file_path):
    """安全读取 wav 时长"""
    try:
        with contextlib.closing(wave.open(file_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)
    except Exception as e:
        print(f"Warning: Could not read duration for {file_path}: {e}")
        return 0.0

def clean_old_files():
    """清理一小时前的临时文件"""
    try:
        cutoff = datetime.now() - timedelta(hours=1)
        for root, _, files in os.walk(UPLOAD_FOLDER):
            for file in files:
                file_path = os.path.join(root, file)
                if datetime.fromtimestamp(os.path.getmtime(file_path)) < cutoff:
                    try: os.remove(file_path)
                    except: pass
    except Exception: pass

# ================= API 核心接口 =================

@app.route('/api', methods=['POST'])
def api():
    if 'audio' not in request.files:
        return jsonify({"code": 1, "msg": "No audio file provided"}), 400
    file = request.files['audio']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"code": 1, "msg": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    # 按日期分文件夹存储
    date_folder = os.path.join(UPLOAD_FOLDER, datetime.now().strftime('%Y-%m-%d'))
    if not os.path.exists(date_folder): os.makedirs(date_folder)
    original_path = os.path.join(date_folder, filename)
    
    temp_files_to_clean = [] # 记录需要清理的中间文件

    try:
        file.save(original_path)
        print(f"\n--- New Task: {filename} ---")
        
        final_subtitles = []

        # >>> 加锁，确保显存独占 <<<
        with gpu_lock:
            # 1. 基础信息获取
            channels = get_audio_channels(original_path)
            hotwords = load_hotwords(HOTWORDS_FILE)
            print(f"   >>> Detected Channels: {channels}")

            # =======================================================
            # 分支 A: 双声道立体声 (Stereo) -> 强制拆分 L/R -> Speaker 0/1
            # =======================================================
            if channels == 2:
                print("   >>> [Mode] Stereo Split (Left=0, Right=1)")
                base_name = os.path.splitext(original_path)[0]
                left_path = f"{base_name}_left.wav"
                right_path = f"{base_name}_right.wav"
                temp_files_to_clean.extend([left_path, right_path])

                # 1.1 使用 ffmpeg 拆分左右声道
                print(">> [1/2] Splitting Channels...")
                # map_channel 0.0.0 = 左, 0.0.1 = 右
                cmd_left = [
                    'ffmpeg', '-y', '-i', original_path, 
                    '-map_channel', '0.0.0', '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', left_path
                ]
                cmd_right = [
                    'ffmpeg', '-y', '-i', original_path, 
                    '-map_channel', '0.0.1', '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', right_path
                ]
                
                # 执行命令
                subprocess.run(cmd_left, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(cmd_right, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # 1.2 识别处理
                if asr_model:
                    print(">> [2/2] FunASR Stereo Transcription (GPU)...")
                    move_funasr(asr_model, "cuda") # 搬运模型到 GPU
                    torch.cuda.synchronize()

                    # 定义任务：左声道固定 speaker=0, 右声道固定 speaker=1
                    tasks = [
                        {"path": left_path, "speaker": 0},
                        {"path": right_path, "speaker": 1}
                    ]

                    try:
                        all_segments = []
                        for task in tasks:
                            t_path = task["path"]
                            t_speaker = task["speaker"]
                            
                            # 获取时长并计算动态 Batch
                            duration = get_wav_duration(t_path)
                            dynamic_batch = calculate_dynamic_batch_size(duration)
                            print(f"      -> Processing Speaker {t_speaker} ({duration:.2f}s, Batch: {dynamic_batch})")

                            res = asr_model.generate(
                                input=t_path,
                                return_raw_text=True,
                                is_final=True,
                                sentence_timestamp=True,
                                batch_size_s=dynamic_batch, # 应用动态 Batch
                                hotword=hotwords,
                                device="cuda",
                                vad_kwargs={
                                    "max_single_segment_time": 60000,
                                    "vad_tail_silence_time": 800,
                                }
                            )

                            if res and 'sentence_info' in res[0]:
                                for item in res[0]['sentence_info']:
                                    start_s = item['timestamp'][0][0] / 1000.0
                                    end_s = item['timestamp'][-1][1] / 1000.0
                                    all_segments.append({
                                        "speaker": t_speaker, # 整数 0 或 1
                                        "start_time": start_s,
                                        "end_time": end_s,
                                        "text": item['text']
                                    })
                        
                        # 按时间顺序合并两个声道的结果
                        all_segments.sort(key=lambda x: x['start_time'])
                        for i, seg in enumerate(all_segments):
                            seg['line'] = i + 1
                            final_subtitles.append(seg)

                    finally:
                        # 用完放回 CPU，防止占用 Pyannote 显存
                        move_funasr(asr_model, "cpu")
                        clear_gpu()

            # =======================================================
            # 分支 B: 单声道/其他 (Mono) -> Pyannote 分离 -> Speaker 0/1/2...
            # =======================================================
            else:
                print("   >>> [Mode] Mono Diarization (Pyannote -> FunASR)")
                process_file_path = os.path.join(date_folder, f"proc_{filename}.wav")
                abs_path = os.path.abspath(process_file_path)
                temp_files_to_clean.append(process_file_path)
                
                # 2.1 预处理：转为 16k mono 并增加静音填充 (Padding)
                # Padding 有助于防止结尾的语音被 VAD 截断
                PADDING_DURATION = 5.0 
                cmd = [
                    'ffmpeg', '-y', '-i', original_path,
                    '-ar', '16000', '-ac', '1',
                    f'-af', f'loudnorm=I=-16:TP=-1.5:LRA=11,apad=pad_dur={PADDING_DURATION}',
                    '-c:a', 'pcm_s16le',
                    abs_path
                ]
                
                print(">> [1/3] Preprocessing Audio (FFmpeg)...")
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # 等待文件系统同步
                time.sleep(0.5) 
                
                total_wav_duration = get_wav_duration(abs_path)

                # 2.2 Pyannote 声纹分离
                diarization_segments = []
                if diarization_pipeline:
                    print(">> [2/3] Pyannote Diarization (GPU)...")
                    try:
                        clear_gpu()
                        # Pyannote 推理
                        diarization = diarization_pipeline(abs_path)
                        for turn, _, speaker in diarization.itertracks(yield_label=True):
                            diarization_segments.append({
                                "start": turn.start, "end": turn.end, "speaker": speaker
                            })
                    except Exception as e:
                        print(f"Pyannote Error: {e}")
                    clear_gpu()

                # 2.3 FunASR 识别与映射
                # 映射表：将 "SPEAKER_00" 映射为 0, "SPEAKER_01" 映射为 1
                speaker_map = {} 
                next_speaker_idx = 0
                
                def get_display_speaker_int(raw_name):
                    nonlocal next_speaker_idx
                    if raw_name not in speaker_map:
                        speaker_map[raw_name] = next_speaker_idx # 赋值整数
                        next_speaker_idx += 1
                    return speaker_map[raw_name]

                if asr_model:
                    print(">> [3/3] FunASR Transcribing (GPU)...")
                    move_funasr(asr_model, "cuda")
                    torch.cuda.synchronize()
                    
                    dynamic_batch_size = calculate_dynamic_batch_size(total_wav_duration)
                    print(f"   >>> [Dynamic Batching] Total: {total_wav_duration:.2f}s | Batch Size: {dynamic_batch_size}")

                    try:
                        res = asr_model.generate(
                            input=process_file_path,
                            return_raw_text=True,
                            is_final=True,
                            sentence_timestamp=True,
                            batch_size_s=dynamic_batch_size, 
                            hotword=hotwords,
                            device="cuda",
                            vad_kwargs={
                                "max_single_segment_time": 60000,
                                "vad_tail_silence_time": 800,
                            }
                        )
                        
                        full_text_str = ""
                        last_valid_raw_speaker = "Unknown" 
                        last_end_time = 0.0

                        if res and 'sentence_info' in res[0]:
                            for i, item in enumerate(res[0]['sentence_info']):
                                start_s = item['timestamp'][0][0] / 1000.0
                                end_s = item['timestamp'][-1][1] / 1000.0
                                
                                # 找到当前时间段的原始 speaker 标签
                                raw_speaker = match_speaker(start_s, end_s, diarization_segments, last_valid_raw_speaker)
                                # 转换为整数 ID
                                display_speaker_int = get_display_speaker_int(raw_speaker)
                                
                                last_valid_raw_speaker = raw_speaker
                                last_end_time = end_s
                                
                                final_subtitles.append({
                                    "line": len(final_subtitles) + 1, 
                                    "speaker": display_speaker_int, # 整数输出
                                    "start_time": start_s, 
                                    "end_time": end_s, 
                                    "text": item['text']
                                })
                                full_text_str += item['text']

                        # --- 防吞字补丁 (处理结果文本比识别文本短的情况) ---
                        if res and 'text' in res[0]:
                            raw_text = res[0]['text'].replace(" ", "")
                            processed_text = full_text_str.replace(" ", "")
                            if len(raw_text) > len(processed_text) + 2:
                                print(f">>> PATCHING! Gap: {len(raw_text) - len(processed_text)}")
                                missing_part = res[0]['text'][len(full_text_str):]
                                if missing_part.strip():
                                    final_subtitles.append({
                                        "line": len(final_subtitles) + 1,
                                        "speaker": get_display_speaker_int(last_valid_raw_speaker),
                                        "start_time": last_end_time,
                                        "end_time": last_end_time + 2.0,
                                        "text": missing_part.strip()
                                    })
                        # -----------------------------------------------

                    except Exception as e:
                        print(f"FunASR Error: {e}")
                        traceback.print_exc()
                    finally:
                        move_funasr(asr_model, "cpu")
                        clear_gpu()

        # ================= 清理与输出 =================
        # 删除生成的 wav 临时文件
        for tmp in temp_files_to_clean:
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except: pass
        # 清理旧文件
        clean_old_files()

        # 控制台打印结果
        print("\n" + "="*50)
        print(f"Result for: {filename} (Total: {len(final_subtitles)})")
        print("-" * 50)
        for item in final_subtitles:
            # 这里的 item['speaker'] 是整数 0, 1, 2...
            print(f"[{item['start_time']:>7.2f}] Speaker {item['speaker']}: {item['text']}")
        print("="*50 + "\n")

        return jsonify({"code": 0, "msg": 'ok', "data": final_subtitles})

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        # 出错时确保模型回收到 CPU，防止影响下一次请求
        if asr_model: move_funasr(asr_model, "cpu")
        clear_gpu()
        return jsonify({"code": 2, "msg": str(e)}), 500

if __name__ == '__main__':
    # 启动服务
    print("Server starting at http://0.0.0.0:9933")
    app.run(debug=False, host="0.0.0.0", port=9933, threaded=True)
