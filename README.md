# stt_api
基于funasr，使用paraformer模型的录音转文字api接口

### 说明
模型会自动下，支持热词功能，GTX1660S流畅跑，20分钟的音频大概10s不到处理完

## 使用方法
### 1. 安装python3.12
### 2. 安装依赖
  ```pip install -r requirements.txt```
### 3. 启动服务
  ```python start.py```
#### 接口使用示例
> curl -X POST http://{server_ip}:9933/api \
  -F "audio=@abc.wav" \
  -H "Content-Type: multipart/form-data"
