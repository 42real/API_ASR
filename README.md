# ASR Service

基于 FastAPI 的语音识别（ASR）REST 服务，实现 `asr-api-spec.md` 中的接口规范。

## 快速开始

```bash
# 安装依赖
uv sync --python 3.12

# 启动服务
uv run uvicorn src.asr_service.main:app --reload --port 8014
```

## 接口

- `POST /asr/start` 启动监听
- `POST /asr/stop` 停止监听并返回识别文本
- `GET /asr/status` 查询是否在监听

## 说明

- 服务依赖麦克风输入（16kHz、单声道、16bit PCM）。
- 同一时刻只允许一个监听会话。
- 识别能力来自 FunASR 模型与现有声纹/VAD 逻辑。
