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

## 技术路线

来源于 `robot-dialogue` 项目的 ASR 实现：

- **ASR**：FunASR `paraformer-zh-streaming`（流式识别）
- **VAD**：FunASR `fsmn-vad`（语音活动检测）
- **Speaker**：FunASR `cam++`（声纹识别，区分老师/学生）
- **Punctuation**：FunASR `ct-punc`（标点恢复）


## 依赖与仓库

核心依赖：
- `funasr`
- `torch`, `torchaudio`, `torchvision`
- `pyaudio`, `numpy`, `scipy`
- `fastapi`, `uvicorn`, `pydantic`, `httpx`, `pytest`

PyTorch 下载源与版本：
- 使用 `pytorch-cu124` 索引（见 `pyproject.toml`）
- 版本固定：`torch==2.6.0`, `torchaudio==2.6.0`, `torchvision==0.21.0`

参考仓库：
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [robot-dialogue](https://github.com/GZhu-Embodied-AI-Lab/robot-dialogue)
## 模型下载

首次启动 ASR 时会自动从 ModelScope 下载模型到本地缓存：

- ASR：`iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online`
- VAD：`iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`
- Speaker：`iic/speech_campplus_sv_zh-cn_16k-common`
- Punctuation：`iic/punc_ct-transformer_cn-en-common-vocab471067-large`

缓存目录默认在：
- `C:\Users\<user>\.cache\modelscope\hub\models\...`
或在`./src/asr_service/asr_core/main.py`中进行修改

## 快速开始

```bash
# 安装依赖
uv sync --python 3.12

# 启动服务
uv run uvicorn src.asr_service.main:app --reload --port 8014
```

## API接口

- `POST /asr/start` 启动监听
- `POST /asr/stop` 停止监听并返回识别文本
- `GET /asr/status` 查询是否在监听

## 接入使用

目前使用 **UDP 流**：

- **UDP 流（机器人方法）**
  - `stream2text_udp(...)`，通过 UDP 音频流输入识别
- **输入**：16kHz / 16bit / 单声道 PCM 流
启动监听：
```bash
curl -X POST -Uri http://127.0.0.1:8014/asr/start -Headers @{ "Content-Type" = "application/json" } -Body "{}"
```
停止监听：
```bash
curl -X POST -Uri http://127.0.0.1:8014/asr/stop -Headers @{ "Content-Type" = "application/json" } -Body "{}"
```
获取状态：
```bash
curl -X GET http://localhost:8014/asr/status
```

具体代码见 `src/asr_service/asr_engine.py` 

## 测试（UDP 流）

按 tutorial 的 pytest 流程：

```bash
uv run pytest -s -k udp_stream
```

说明：
- 需要准备 `./tests/test.wav`（16kHz、单声道、16bit WAV）
- 测试流程：`/asr/start` → UDP 推流（循环播放 5 秒）→ `/asr/stop` → `/asr/status`
- 识别结果会在终端打印

## 运行限制与注意事项

- **单会话**：同一时间只允许一个监听会话。
- **网络依赖**：模型首次下载需要可访问 ModelScope。
- **音频格式**：必须是 16kHz/16bit/单声道 PCM 流。
- **识别模式**：目前识别模式为plain，识别并返回包含识别对象的列表识别结果。另一个dialog模式需要老师角色触发关键词以开始/停止记录。duration是识别时长，目前设为None为持续识别。
- **声纹注册**：若未提供老师声纹文件，系统会将所有说话人视为学生。
  - 配置路径见 `src/asr_service/asr_core/config.py`

## 故障排查

- 模型下载失败：检查网络连接或代理配置。
