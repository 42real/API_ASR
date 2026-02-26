import os
import sys
import traceback

# 优化路径处理
current_dir = os.path.dirname(os.path.abspath(__file__))
sub_dir = os.path.join(current_dir, 'realtime_meeting_assistant')
sys.path.insert(0, sub_dir)  # 优先使用子目录

from .realtime_meeting_assistant.main import RealtimeAssistant

class SpeakerAudio:
    """
    实时会议助手接口。
    将所有逻辑委托给 realtime_meeting_assistant/main.py 中的 RealtimeAssistant 类。
    """
    def __init__(self):
        """初始化SpeakerAudio接口"""
        print("正在初始化 SpeakerAudio 接口...")
        try:
            self.assistant = RealtimeAssistant()
        except Exception as e:
            print(f"初始化失败: {e}")
            raise

    def process_audio_stream(self, audio_stream, mode: str = "plain") -> list:
        """
        处理音频流并返回识别结果
        
        Args:
            audio_stream: 生成16bit pcm音频数据的生成器
            mode: 模式选择，"plain"=普通ASR，"dialog"=启用开始/停止指令
            
        Returns:
            list: 识别结果列表，每个结果包含说话人、文本等信息
            
        Raises:
            Exception: 处理过程中发生的任何异常
        """
        print("正在通过接口处理音频流...")
        try:
            return self.assistant.run_stream(audio_stream, mode=mode)
        except Exception as e:
            print(f"音频流处理失败: {e}")
            traceback.print_exc()
            raise

if __name__ == "__main__":
    try:
        audio_processor = SpeakerAudio()
        # 这里需要提供一个音频流生成器，示例中使用麦克风
        from realtime_meeting_assistant.main import MicrophoneStream
        results = audio_processor.process_audio_stream(MicrophoneStream())
        
        if results:
            print("\n=== 最终识别结果 ===")
            for i, result in enumerate(results, 1):
                speaker = result.get('speaker', 'Unknown')
                text = result.get('text', '')
                print(f"{i}. {speaker}: {text}")
    except Exception as e:
        print(f"主程序错误: {e}")
        sys.exit(1)