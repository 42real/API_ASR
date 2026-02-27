import os
import sys
import traceback

# Ensure asr_core is on sys.path for legacy imports inside it
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(current_dir, "asr_core")
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)

from .asr_core.main import RealtimeAssistant


class SpeakerAudio:
    """
    ASR 适配层：封装 RealtimeAssistant，对外只暴露 process_audio_stream。
    """

    def __init__(self):
        """初始化 SpeakerAudio 接口"""
        print("正在初始化 SpeakerAudio 接口...")
        try:
            self.assistant = RealtimeAssistant()
        except Exception as e:
            print(f"初始化失败: {e}")
            raise

    def process_audio_stream(self, audio_stream, mode: str = "plain") -> list:
        """
        处理音频流并返回识别结果。

        Args:
            audio_stream: 生成 16bit PCM 音频数据的生成器
            mode: 模式选择，"plain"=普通ASR，"dialog"=启用开始/停止指令

        Returns:
            list: 识别结果列表（包含说话人、文本等字段）
        """
        print("通过接口处理音频流中...")
        try:
            return self.assistant.run_stream(audio_stream, mode=mode)
        except Exception as e:
            print(f"音频流处理失败: {e}")
            traceback.print_exc()
            raise


if __name__ == "__main__":
    # 本文件不建议直接运行，保留调试入口
    raise SystemExit("Please run via the ASR service, not speaker_audio.py directly.")
