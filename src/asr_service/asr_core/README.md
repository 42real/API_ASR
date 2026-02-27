# 实时多人语音识别会议助手

基于 [FunASR](https://github.com/alibaba-damo-academy/FunASR) 的实时语音识别与说话人识别系统，支持“老师/学生”身份区分、模糊指令识别，以及更稳健的声纹管理。

## 特性

- **实时字幕**：`Paraformer-zh-streaming` 流式识别，低延迟上屏。
- **精准端点**：`FSMN-VAD` 做语音活动检测，避免截断或拖尾。
- **说话人识别**：`Cam++` 声纹识别区分不同说话人。
- **老师声纹库（画廊注册）**：从一段老师音频切片提取多段嵌入，增强对不同音色/语调的适配。
- **在线学习**：会话中对老师嵌入做加权更新，逐步提升识别稳定性。
- **模糊指令识别**：支持识别“停止记录”“下课”等指令，兼容“下 客/下客”等轻微误识。
- **调试显示**：终端行内显示当前说话人及识别文本，支持分数提示（如 `T:0.xx|S:0.xx`）。

## 环境要求

- Windows / Linux / macOS
- Python 3.8+
- 麦克风设备

## 安装与模型

1. **安装依赖**（示例）
    ```bash
    pip install funasr pyaudio numpy
    ```
    - Windows 安装 `pyaudio` 若失败，可使用 `pipwin install pyaudio` 或下载安装匹配的 `.whl`。

2. **模型下载**
    首次运行时 FunASR 将自动从 ModelScope 下载 `ASR / VAD / Speaker` 预训练模型，需保证网络可用。

## 使用方法

推荐入口在项目根目录：

```bash
python speaker_audio.py
```

该入口会通过接口类 `SpeakerAudio` 调用子模块中的 `RealtimeAssistant` 完整逻辑。

也可在子模块中进行单元测试：

```bash
python realtime_meeting_assistant/main.py
```

启动后初始化模型（首次可能较慢），随后在终端行内显示：
- 当前说话人（如 `Teacher` / `Student`）
- 流式识别文本（持续追加）
- 当 VAD 判定句子结束时，打印最终文本并换行

## 配置说明

请在配置文件中调整参数：[realtime_meeting_assistant/config.py](realtime_meeting_assistant/config.py)

- **`SIMILARITY_THRESHOLD`**：声纹相似度阈值。当前为增强老师识别而降低到约 `0.32`。
- **`TEACHER_WAV_PATH`**：用于老师声纹“画廊注册”的预置音频路径。
- **`REGISTERED_DB_PATH`**：老师声纹库路径，已迁移至子目录 [realtime_meeting_assistant/teacher_db/teacher_db.pkl](realtime_meeting_assistant/teacher_db/teacher_db.pkl)。
- **音频参数**：`SAMPLE_RATE / FORMAT / CHANNELS / VAD_CHUNK_SIZE / ASR_CHUNK_SIZE / VAD_CHUNK_DURATION_MS`。

老师注册支持两种方式：
- 放置预置音频到 `TEACHER_WAV_PATH`，程序自启动时切片提取多段嵌入并注册；
- 已有 `teacher_db.pkl` 时将自动加载（无需重复注册）。

## 架构说明

- **接口层**：[speaker_audio.py](speaker_audio.py)：用户入口。初始化并调用 `RealtimeAssistant.run()`。
- **核心逻辑**：[realtime_meeting_assistant/main.py](realtime_meeting_assistant/main.py)：`RealtimeAssistant` 封装 ASR/VAD/声纹与会话状态。
- **声纹管理**：[realtime_meeting_assistant/speaker_manager.py](realtime_meeting_assistant/speaker_manager.py)：老师/学生识别、在线学习（加权更新）。
- **工具方法**：[realtime_meeting_assistant/utils.py](realtime_meeting_assistant/utils.py)：注册（多段切片）、临时 WAV 保存、指令检测（模糊匹配）。

运行时数据流：
1. **VAD**：每 200ms 检测语音开始/结束，触发会话状态切换；
2. **ASR**：累计约 600ms 音频做流式识别，行内增量上屏；
3. **声纹**：约 1.2s 音频切片做嵌入提取，匹配老师/学生；
4. **指令**：当当前说话人是老师时，对整句做模糊匹配（“停止记录”“下课”等），收到后安全退出。

## 常见问题

- **`pyaudio` 安装失败**：Windows 建议使用 `pipwin install pyaudio` 或安装匹配的 `.whl`。
- **首次运行较慢**：模型自动下载，取决于网络与磁盘。
- **老师识别不稳定**：检查是否已进行“画廊注册”；必要时降低阈值或提供更清晰的样本。
- **指令未触发**：确保当前识别说话人是老师，并清晰说出“停止记录/下课”。
