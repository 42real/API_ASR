import pyaudio

# Audio Configuration
# VAD 使用 200ms 切片，ASR 使用 600ms 切片
VAD_CHUNK_DURATION_MS = 200
ASR_CHUNK_DURATION_MS = 600

SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16

# 计算对应的采样点数
VAD_CHUNK_SIZE = int(SAMPLE_RATE * VAD_CHUNK_DURATION_MS / 1000)
ASR_CHUNK_SIZE = int(SAMPLE_RATE * ASR_CHUNK_DURATION_MS / 1000)

# 兼容旧代码，默认 CHUNK_SIZE 指向 VAD 的大小（因为我们是按 VAD 粒度读取的）
CHUNK_SIZE = VAD_CHUNK_SIZE 

# Speaker Configuration
# 激进调整：降低到 0.32，优先保证老师能被认出来
SIMILARITY_THRESHOLD = 0.45
REGISTERED_DB_PATH = "./src/asr_service/asr_core/teacher_db/teacher_db.pkl"
# 如果此文件存在，将优先使用此文件进行注册，而不是录音
# TEACHER_WAV_PATH = "realtime_meeting_assistant/teacher_audio/teacher_reg.wav"
TEACHER_WAV_PATH = "./src/asr_service/asr_core/teacher_audio/teacher_register.wav"

# File Paths
TEMP_WAV_PATH = "temp_chunk.wav"

# Commands
# 扩充指令库，包含常见的口语表达

COMMAND_KEYWORDS_STOP = [
    # 停止/下课指令
    # "停止记录", "结束会议", "停止录音", "关掉录音",
    # "下课", "现在下课", "下课了", "好了下课",
    "老师"
]

COMMAND_KEYWORDS_START = [
    # 开始/上课指令
    # "上课", "开始上课", "现在上课", "同学们上课",
    "请问"
]

# 所有指令的检测
COMMAND_KEYWORDS = COMMAND_KEYWORDS_STOP + COMMAND_KEYWORDS_START

# Role and command registry (permission model)
ROLE_TEACHER = "teacher"
ROLE_STUDENT = "student"
ROLE_UNKNOWN = "unknown"

COMMAND_DEFINITIONS = [
    {
        "id": "start_session",
        "type": "start",
        "keywords": COMMAND_KEYWORDS_START,
        "roles": [ROLE_STUDENT, ROLE_TEACHER],
    },
    {
        "id": "stop_session",
        "type": "stop",
        "keywords": COMMAND_KEYWORDS_STOP,
        "roles": [ROLE_STUDENT, ROLE_TEACHER],
    },
]
