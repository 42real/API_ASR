import sys
from pathlib import Path

# 添加项目根目录到模块搜索路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from .audio_generate.speaker_audio import SpeakerAudio#语音识别2
from shared_package import ppt2audio
from shared_package import audio_tools
from shared_package.ppt2audio import PPT2AudioTools
import os
import librosa
import time
from pathlib import Path
import logging
from time import sleep
import socket
import time
import threading
import struct
import wave
import socket
import numpy as np
import struct
import time
import os
from shared_package.tts import TTS

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)8s: %(asctime)s [%(filename)s:%(lineno)4d - %(funcName)20s() ]: %(message)s",
)
logger = logging.getLogger(__name__)

VAD_CHUNK_SIZE = 3200  # 200ms of 16kHz audio (16000 samples/sec * 0.2 sec)

import librosa


class Audio:
    """
    PPT转语音处理类
    功能: 将PPT文件转换为语音文件，并提供播放和时长查询功能
    """
    def __init__(self, wav_file: str = None):
        """
        初始化Audio类
        
        Args:
            wav_file: str, 可选的WAV音频文件路径，用于声音克隆
        """
        self.ppt2audio = PPT2AudioTools()
        self.audio_tools = audio_tools
        self.audioClient = None  # 播放器
        self.tts = TTS()#语音合成
        self.audio_processor = SpeakerAudio()#语音识别2
        self.voice_clone_wav = wav_file  # 声音克隆用的参考音频
        self.wav_text=None
        if wav_file:
            self.wav_text=self.wav2text(wav_file)
    def text2wav(self, text: str, dest_wav: str, use_voice_clone: bool = True):
        """
        文字转语音（支持声音克隆）

        Args:
            text: str, 讲话内容
            dest_wav: 输出的wav文件路径
            use_voice_clone: bool, 是否使用声音克隆，默认为True
        """
        if use_voice_clone:
            self.tts.tts(text=text, dest_wav=dest_wav, prompt_wav=self.voice_clone_wav, prompt_text=self.wav_text)
        else:
            self.tts.tts(text=text, dest_wav=dest_wav, prompt_wav=None, prompt_text=None)
        print("语音生成完成")

#流式输出demo
    def text2wav_stream(self, text: str, dest_wav: str, use_voice_clone: bool = True):
        """
        文字转语音流式输出版（支持声音克隆）

        Args:
            text: str, 讲话内容
            dest_wav: 输出的wav文件路径
            use_voice_clone: bool, 是否使用声音克隆，默认为True
        """
        if use_voice_clone:
            self.tts.tts_stream(text=text,dest_wav=dest_wav, prompt_wav=self.voice_clone_wav, prompt_text=self.wav_text)
        else:
            self.tts.tts_stream(text=text,dest_wav=dest_wav, prompt_wav=None, prompt_text=None)
        print("语音生成完成")

    def wav2text(self, wav_file: str, mode: str = "plain") -> str:
        """
        语音转文字

        Args:
            wav文件路径，格式：s16l, 16kHz, 1 channel
            mode: 模式选择，"plain"=普通ASR，"dialog"=启用开始/停止指令

        Returns:
            str: 识别内容
        """
# #语音识别2
        # 验证文件存在
        if not os.path.exists(wav_file):
            raise FileNotFoundError(f"WAV file not found: {wav_file}")
        wav_file = self.audio_tools.resample_audio(wav_file)
        # 读取WAV文件并转换为音频流生成器
        def wav_audio_stream_generator():
            with wave.open(wav_file, 'rb') as wav:
                # 验证WAV格式
                if wav.getnchannels() != 1:
                    raise ValueError("WAV file must be mono (1 channel)")
                if wav.getframerate() != 16000:
                    raise ValueError("WAV file must have 16kHz sample rate")
                if wav.getsampwidth() != 2:
                    raise ValueError("WAV file must have 16-bit samples")
                
                # 读取所有音频数据
                frames = wav.readframes(wav.getnframes())
                
                # 按块读取并生成音频流（与VAD_CHUNK_SIZE保持一致）
                chunk_size = VAD_CHUNK_SIZE * 2  # VAD_CHUNK_SIZE是采样点数，乘以2是字节数
                offset = 0
                
                while offset < len(frames):
                    chunk = frames[offset:offset + chunk_size]
                    if len(chunk) > 0:
                        yield chunk
                    offset += chunk_size
        
        # 使用现有的SpeakerAudio类处理音频流
        results = self.audio_processor.process_audio_stream(wav_audio_stream_generator(), mode=mode)
        
        # 拼接识别结果
        final_text = ' '.join([res['text'] for res in results if 'text' in res])
        
        return final_text


    def _wav2stream(
        self, wav_file: str, udp_address: str = "239.168.123.161:5555"
    ) -> None:
        """
        (测试用)将wav文件转成UDP流，无限循环播放

        音频流数据格式为单通道/16K采样率/16bit

        Args:
            wav_file: 音频文件
            udp_address: udp流地址
        """

        # 验证文件是否存在
        if not os.path.exists(wav_file):
            raise FileNotFoundError(f"WAV file not found: {wav_file}")
        wav_file = self.audio_tools.resample_audio(wav_file)
        # 解析UDP地址
        try:
            udp_host, udp_port = udp_address.split(':')
            udp_port = int(udp_port)
        except ValueError:
            raise ValueError("Invalid UDP address format. Expected format: 'ip:port'")
        
        # 打开WAV文件
        with wave.open(wav_file, 'rb') as wav:
            # 获取WAV文件参数
            n_channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            n_frames = wav.getnframes()
            
            print(f"Original WAV format: {n_channels} channels, {sample_rate}Hz, {sample_width*8}bit")
            
            # 检查是否需要格式转换
            needs_conversion = False
            if n_channels != 1 or sample_rate != 16000 or sample_width != 2:
                print("Format conversion needed. Target: mono, 16000Hz, 16bit")
            
            # 读取所有音频数据
            frames = wav.readframes(n_frames)
            
            # 如果需要格式转换
            if needs_conversion:
                # 将原始数据转换为numpy数组
                if sample_width == 1:  # 8bit
                    audio_data = np.frombuffer(frames, dtype=np.uint8)
                    audio_data = audio_data.astype(np.int16) * 256  # 8bit to 16bit
                elif sample_width == 2:  # 16bit
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                elif sample_width == 3:  # 24bit
                    # 24bit to 16bit conversion
                    audio_data = np.zeros(n_frames * n_channels, dtype=np.int16)
                    for i in range(n_frames * n_channels):
                        start_idx = i * 3
                        # 取高16位
                        audio_data[i] = int.from_bytes(
                            frames[start_idx:start_idx+2], 
                            byteorder='little', 
                            signed=True
                        )
                elif sample_width == 4:  # 32bit
                    audio_data = np.frombuffer(frames, dtype=np.int32)
                    audio_data = (audio_data >> 16).astype(np.int16)  # 32bit to 16bit
                
                # 重塑为正确的形状 (frames, channels)
                audio_data = audio_data.reshape(-1, n_channels)
                
                # # 转换为单声道（如果需要）
                # if n_channels > 1:
                #     print(f"Converting {n_channels} channels to mono...")
                #     mono_audio = np.mean(audio_data, axis=1).astype(np.int16)
                # else:
                #     mono_audio = audio_data.flatten()
                
                # # 重采样到16kHz（如果需要）
                # if sample_rate != 16000:
                #     print(f"Resampling from {sample_rate}Hz to 16000Hz...")
                #     mono_audio = self._resample_audio(mono_audio, sample_rate, 16000)
                
                # # 确保数据类型为int16
                # audio_data = mono_audio.astype(np.int16)
                # audio_bytes = audio_data.tobytes()
            else:
                # 格式已经符合要求
                audio_bytes = frames
            
            print(f"Audio data prepared. Total bytes: {len(audio_bytes)}")
            
            # 创建UDP socket
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                print(f"Created UDP socket. Sending to {udp_host}:{udp_port}")
                
                # 无限循环播放
                chunk_size = 320  # 10ms of 16kHz mono 16bit audio = 16000 * 0.01 * 2 = 320 bytes
                total_chunks = len(audio_bytes) // chunk_size
                
                print(f"Starting infinite loop playback. Chunk size: {chunk_size} bytes")
                print(f"Press Ctrl+C to stop...")
                
                while True:
                    for i in range(total_chunks):
                        start_idx = i * chunk_size
                        end_idx = start_idx + chunk_size
                        if end_idx > len(audio_bytes):
                            break
                        
                        chunk = audio_bytes[start_idx:end_idx]
                        # print(f"Sending chunk {i+1}/{total_chunks}...")
                        try:
                            sock.sendto(chunk, (udp_host, udp_port))
                        except Exception as e:
                            print(f"Error sending UDP packet: {e}")
                            time.sleep(0.1)  # 短暂等待后重试
                            continue
                        
                        # 控制发送速率，保持16kHz采样率
                        time.sleep(chunk_size / (16000 * 2))  # 2 bytes per sample
                        
            except KeyboardInterrupt:
                print("\nPlayback stopped by user")
            except Exception as e:
                print(f"Error during UDP streaming: {e}")
            finally:
                # 关闭socket
                if 'sock' in locals():
                    sock.close()
                print("UDP socket closed")

    def stream2text(
        self, udp_address: str = "239.168.123.161:5555", duration: float = 5.0, mode: str = "plain"
    
    ) -> str:
        """
        将UDP音频流转换成文字

        音频流数据格式为单通道/16K采样率/16bit

        Args:
            udp_address: udp地址
            duration: 持续时长(秒)
            mode: 模式选择，"plain"=普通ASR，"dialog"=启用开始/停止指令

        Returns:
            str: 识别内容
        """
#语音识别2
        # 解析UDP地址
        host, port_str = udp_address.split(':')
        port = int(port_str)
        
        # 检查是否为多播地址
        is_multicast = False
        try:
            ip_parts = list(map(int, host.split('.')))
            if len(ip_parts) == 4 and 224 <= ip_parts[0] <= 239:
                is_multicast = True
        except:
            pass

        # 创建UDP socket
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp_socket.settimeout(1.0)  # 1秒超时

        try:
            if is_multicast:
                # 绑定到任意地址，端口
                udp_socket.bind(('', port))
                # 加入多播组
                group = socket.inet_aton(host)
                mreq = struct.pack('4sL', group, socket.INADDR_ANY)
                udp_socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            else:
                udp_socket.bind((host, port))
                
        except Exception as e:
            udp_socket.close()
            raise Exception(f"UDP绑定失败: {e}")

        # 计算VAD_CHUNK_SIZE对应的字节数 (200ms * 16000 * 2 = 6400字节)
        chunk_size_bytes = 6400

        # 新增：UDP音频流生成器（关键修改）
        def udp_audio_stream_generator():
            """将UDP流转换为符合VAD要求的音频块生成器"""
            start_time = time.time()
            buffer = b''  # 用于累积数据
            
            try:
                while time.time() - start_time < duration:
                    try:
                        # 接收UDP数据
                        data, addr = udp_socket.recvfrom(4096)
                        buffer += data
                        
                        # 当累积数据达到VAD要求的块大小时yield
                        while len(buffer) >= chunk_size_bytes:
                            chunk = buffer[:chunk_size_bytes]
                            buffer = buffer[chunk_size_bytes:]
                            yield chunk
                            
                    except socket.timeout:
                        # 超时检查部分数据
                        if len(buffer) >= chunk_size_bytes:
                            chunk = buffer[:chunk_size_bytes]
                            buffer = buffer[chunk_size_bytes:]
                            yield chunk
                        continue
                    except OSError:
                        # socket关闭时退出
                        break
                        
            finally:
                # 清理资源
                if is_multicast:
                    try:
                        group = socket.inet_aton(host)
                        mreq = struct.pack('4sL', group, socket.INADDR_ANY)
                        udp_socket.setsockopt(socket.IPPROTO_IP, socket.IP_DROP_MEMBERSHIP, mreq)
                    except:
                        pass
                udp_socket.close()

        # 使用SpeakerAudio处理UDP流（复用现有逻辑）

        results = self.audio_processor.process_audio_stream(udp_audio_stream_generator(), mode=mode)
        final_text = '\n'.join([f"{res['speaker']}: {res['text']}" for res in results if 'text' in res])
        print("\naudio_processor识别结果：", final_text)
        # # 拼接识别结果
        # final_text = ' '.join([res['text'] for res in results if 'text' in res])
        return final_text



############以下范围为原audio的改版################################
    def ppt2wav(self, ppt_file: str, wav_dir: str, use_voice_clone: bool = True):
        """
        将PPT文件转换为多个WAV音频文件（支持声音克隆）

        包含WAV文件的文件夹，格式为
        {0.wav,1.wav...}    # 从第一张PPT开始的讲稿
        {opening.wav, ending.wav, question.wav} # 开场，谢幕，请提问

        Args:
            ppt_file: PPT文件路径
            wav_dir: 输出的wav文件夹
            use_voice_clone: bool, 是否使用声音克隆，默认为True
        """
        ppt_file= ppt2audio.pptx_to_pdf(ppt_file)# 将pptx转为pdf格式
        # Test Data
        PPT_FILE_PATH = Path(ppt_file).resolve()
        PPT_NAME = PPT_FILE_PATH.stem
        CURRENT_DIR = PPT_FILE_PATH.parent
        PPT_PROCESS_DIR = CURRENT_DIR / PPT_NAME
        
        # 判断是否存在已经预处理好的ppt2wav文件 
        PPT_PROCESS_DIR.mkdir(parents=True, exist_ok=True)

        # 1. pdf格式的ppt切成多张图片
        pages_path_list = self.ppt2audio.ppt_pdf2images(str(PPT_PROCESS_DIR), str(PPT_FILE_PATH))
              
        # 2. PPT（image）语义理解与解析 
        PPT_CONTENT_PATH = PPT_PROCESS_DIR / f"{PPT_NAME}_ppt_content.json"
        self.ppt2audio.image2txt_content(str(PPT_PROCESS_DIR), PPT_NAME, pages_path_list, str(PPT_CONTENT_PATH))
        
        # # 3. 将PPT的内容转化为讲课讲稿
        AUDIO_TXT_PATH = PPT_PROCESS_DIR / f"{PPT_NAME}_ppt_speech_script.json"
        if not AUDIO_TXT_PATH.exists():
            self.ppt2audio.txt_content2script(PPT_NAME, str(PPT_PROCESS_DIR), str(PPT_CONTENT_PATH), str(AUDIO_TXT_PATH))
        
        # 4. 讲课讲稿生成为指定格式的wav音频文件
        WAV_DIR = PPT_PROCESS_DIR / "wav"
        WAV_DIR= wav_dir
        if use_voice_clone:
             self.tts.script2wav(str(AUDIO_TXT_PATH), str(PPT_PROCESS_DIR), str(WAV_DIR), prompt_wav=self.voice_clone_wav, prompt_text=self.wav_text)
        else:
            self.tts.script2wav(str(AUDIO_TXT_PATH), str(PPT_PROCESS_DIR), str(WAV_DIR), prompt_wav=None, prompt_text=None)
    def wav_length(self, wav_file: str) -> float:
        """
        获取WAV音频文件的时长

        Args:
            wav_file: WAV文件路径

        Return:
            音频时长(秒)
        """
        # 加载音频文件，保持原始采样率
        audio, original_sr = librosa.load(wav_file, sr=None) 
        sec_time = len(audio) / original_sr
        
        return sec_time

############以上范围为原audio的改版################################

    def close(self) -> None:
        """
        关闭音频连接
        """
        # if self.audioClient:
        #     self.audioClient.PlayStop("example") 
        # else:
        pass

# 测试代码
if __name__ == "__main__":
    # init
    logger.info("init")
    audio = Audio("./Shenteng1.wav")
    # text2wav
    wav_path = "./test.wav"
    audio.text2wav("hello world", wav_path)
    logger.info(f"output wav: {wav_path}")
    # wav2text
    asr_result = audio.wav2text(wav_path)
    logger.info(f"asr result: {asr_result}")

    # ppt2wav
    ppt_path = "./test.pptx"
    wav_dir = "./wav/"
    audio.ppt2wav(ppt_path, wav_dir)

    # wav_length
    duration = audio.wav_length(wav_path)
    logger.info(f"length of {wav_path}: {duration}s")
    audio.close()

    # wav2stream, run in another thread
    t1 = threading.Thread(target=audio._wav2stream, daemon=True, args=([wav_path]))
    t1.start()
    sleep(1)
    # stream2text
    asr_result = audio.stream2text()
    logger.info(f"asr result: {asr_result}")
    print(f"asr result: {asr_result}")
    t1.join(timeout=5.0)

    # # 从麦克风获取音频
    # results = audio.audio_processor.process_audio_stream(
    #     audio_tools.microphone_audio_stream_generator(duration_seconds=None)
    # )
    # # 打印结果
    # print("\n=== 识别结果 ===")
    # for result in results:
    #     print(f"{result['speaker']}: {result['text']}")
    
    # close
    logger.info("closing")
    audio.close()
