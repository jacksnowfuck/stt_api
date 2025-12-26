# Offline ASR & Diarization API (Low-VRAM Optimized)

这是一个基于 **FunASR** (语音转写) 和 **Pyannote** (说话人分离) 的离线语音识别 API 服务。

特别针对 **6GB 显存 (Consumer GPU)** 进行了深度优化，采用 **"Pyannote 驻留 + FunASR 动态轮换"** 的策略，在保证响应速度的同时，防止显存溢出 (OOM)。

## 🚀 核心特性

* **完全离线 (Offline)**：设置了 `HF_HUB_OFFLINE=1`，无需联网即可运行（需提前下载模型）。
* **6GB 显存极致优化**：
* **Pyannote**：常驻显存（约占 1.5GB），消除 10秒+ 的加载延迟。
* **FunASR**：平时驻留内存，推理时瞬间搬入 GPU，用完即走。
* **全局锁 (GPU Lock)**：强制串行处理，防止多请求并发炸显存。


* **自动音频清洗**：内置 `FFmpeg` 预处理，强制将任意音频（MP3/FLAC/WAV）转换为 **16000Hz 单声道**，确保模型识别率达到最佳。
* **支持 FLAC/WAV**：针对电话录音场景优化，推荐使用无损格式。

## 🛠️ 环境依赖

### 1. 系统要求

* **OS**: Linux (推荐) / Windows
* **GPU**: NVIDIA 显卡，显存  6GB (推荐 8GB+)
* **CUDA**: 11.8 或 12.x

### 2. 必须安装 FFmpeg

API 依赖 FFmpeg 进行音频重采样，**必须安装**，否则会报错。

* **Ubuntu/Debian**:
```bash
sudo apt-get update && sudo apt-get install ffmpeg

```


* **CentOS/RHEL**:
```bash
sudo yum install ffmpeg

```


* **Windows**: 下载 FFmpeg 二进制包并将 `bin` 目录加入系统环境变量 `PATH`。

### 3. Python 依赖

建议使用 Python 3.8 - 3.10。

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118  # 根据你的 CUDA 版本调整
pip install flask werkzeug funasr modelscope pyannote.audio

```

## 📥 模型准备 (离线模式)

由于代码强制离线运行，首次运行前请确保模型已下载到本地 HuggingFace/ModelScope 缓存目录，或手动指定路径。

1. **Pyannote 3.1**: `pyannote/speaker-diarization-3.1`
2. **FunASR**: `paraformer-zh` (v2.0.4), `fsmn-vad`, `ct-punc-c`

## 🏃‍♂️ 启动服务

```bash
python app.py

```

* **启动阶段**：控制台会显示 `Moving Pyannote to GPU...`，此时会卡顿约 10-15 秒，显存占用上升，这是正常的初始化过程。
* **就绪状态**：看到 `Server starting at http://0.0.0.0:9934` 即表示服务已就绪。

## 🔌 API 接口说明

### 语音识别与分离

* **URL**: `/api`
* **Method**: `POST`
* **Body**: `form-data`

| Key | Type | Description |
| --- | --- | --- |
| `audio` | File | 音频文件 (支持 `.wav`, `.flac`, `.mp3`, `.m4a`) |

### 示例调用 (cURL)

```bash
curl -X POST -F "audio=@/path/to/record.flac" http://127.0.0.1:9934/api

```

### 返回示例

```json
{
    "code": 0,
    "msg": "ok",
    "data": [
        {
            "line": 1,
            "speaker": "SPEAKER_01",
            "start_time": 0.52,
            "end_time": 2.15,
            "text": "喂，你好，请问是张先生吗？"
        },
        {
            "line": 2,
            "speaker": "SPEAKER_00",
            "start_time": 2.80,
            "end_time": 3.50,
            "text": "对，我是。"
        }
    ]
}

```

## ⚠️ 音频质量规范 (至关重要)

为了获得准确的说话人分离效果，请严格遵守以下录音/压缩标准：

1. **推荐格式**: **FLAC** (无损压缩) 或 **WAV** (无损)。
2. **MP3 禁忌**:
* ❌ **严禁使用 32kbps / 64kbps** 低码率 MP3（会导致严重的人声失真和混淆）。
* ✅ 如必须使用 MP3，码率至少 **128kbps**，推荐 **320kbps**。


3. **采样率**:
* 源文件最好  8000Hz。
* API 内部会自动升频到 16000Hz 处理，但无法修复源文件丢失的高频信息。



## 🧠 显存管理逻辑 (Technical Details)

针对 6GB 显存的调度策略：

1. **常驻层 (Pyannote)**：
* 占用约 1.5GB VRAM。
* 原因：Pyannote 初始化极慢，为了 API 秒级响应，必须常驻。


2. **动态层 (FunASR)**：
* 推理时占用额外约 3GB VRAM。
* 原因：FunASR 初始化快，平时放在 CPU 内存，仅在转写瞬间加载到 GPU。


3. **峰值功耗**：
* 推理期间总占用：`1.5G + 3G ≈ 4.5G` < 6G (安全)。
* 推理结束：立即释放 FunASR，回落至 1.5G。



## 📝 配置文件

* `hotwords.txt`: 在同目录下创建此文件，每行一个热词，可提升特定词汇（如公司名、产品名）的识别准确率。无需重启服务，即时生效。
