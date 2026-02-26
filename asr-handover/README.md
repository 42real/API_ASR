# ASR 代码交接包（robot-dialogue）

本目录为从 `robot-dialogue` 项目中提取的 ASR 相关代码，方便交接与独立阅读。

## 目录结构

```
apiver/asr-handover/
├── audio/
│   ├── audio.py
│   └── audio_generate/
│       ├── speaker_audio.py
│       └── realtime_meeting_assistant/
│           ├── main.py
│           ├── config.py
│           ├── speaker_manager.py
│           ├── utils.py
│           ├── README.md
│           └── requirements.txt
```

## 组件说明

- `audio/audio.py`
  - ASR 的对外入口封装类 `Audio`。
  - 关键方法：
    - `wav2text(wav_file, mode)`：WAV 文件识别。
    - `stream2text(udp_address, duration, mode)`：UDP 流识别。
  - 内部通过 `SpeakerAudio.process_audio_stream` 处理音频流。

- `audio/audio_generate/speaker_audio.py`
  - ASR 适配层，封装 `RealtimeAssistant`，对外只暴露 `process_audio_stream`。

- `audio/audio_generate/realtime_meeting_assistant/main.py`
  - ASR/VAD/声纹识别主流程。
  - 使用 FunASR 模型：
    - ASR：`paraformer-zh-streaming`
    - VAD：`fsmn-vad`
    - 声纹：`cam++`
    - 标点：`ct-punc`
  - 提供 `RealtimeAssistant.run_stream(audio_stream, mode)`（核心）

- `audio/audio_generate/realtime_meeting_assistant/config.py`
  - 采样率、chunk 大小、命令词、阈值等配置。

- `audio/audio_generate/realtime_meeting_assistant/speaker_manager.py`
  - 声纹管理、老师/学生识别、在线学习策略。

- `audio/audio_generate/realtime_meeting_assistant/utils.py`
  - 教师声纹注册、指令检测、临时 wav 保存等工具函数。

## 运行方式（示例）

1. 安装依赖（参考 `requirements.txt`）
2. 直接运行：

```bash
python audio/audio_generate/realtime_meeting_assistant/main.py
```

3. 代码调用（示例）：

```python
from audio.audio import Audio

audio = Audio()
text = audio.wav2text("/path/to.wav", mode="plain")
print(text)
```

## 说明

- 该交接包保持原始代码结构与内容，仅做“复制提取”，未进行逻辑改动。
- 如果需要进一步精简依赖或重构 API，请告知目标使用场景。
