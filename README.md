# ASR Service

基于 FastAPI 的语音识别（ASR）REST 服务，实现 `asr-api-spec.md` 中的接口规范。

## 项目结构

```
API_ASR/
├── pyproject.toml
├── README.md
├── src/asr_service/
│   ├── main.py              # FastAPI 应用
│   ├── asr_engine.py        # 会话管理 / 音频入口
│   ├── speaker_audio.py     # ASR 适配层
│   └── asr_core/            # FunASR + VAD + 声纹识别核心逻辑
└── tests/
    └── test_udp_stream.py   # UDP 音频流集成测试
```

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

## 入口模式（asr_engine.py 中切换）

目前默认使用 **UDP 流**：

- **UDP 流（机器人方法）**
  - `stream2text_udp(...)`，通过 UDP 音频流输入识别

## 测试（UDP 流）

测试方式按 tutorial 的 pytest 流程：

```bash
uv run pytest -s -k udp_stream
```

说明：
- 需要准备 `./tests/test.wav`（16kHz、单声道、16bit WAV）
- 测试流程：`/asr/start` → UDP 推流（循环播放 5 秒）→ `/asr/stop` → `/asr/status`
- 识别结果会在终端打印

## 说明

- 同一时刻只允许一个监听会话。
- 识别能力来自 FunASR 模型与现有声纹/VAD 逻辑。
