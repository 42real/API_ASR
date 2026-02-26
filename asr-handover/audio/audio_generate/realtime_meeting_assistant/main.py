import os
import time
import traceback
import numpy as np
import pyaudio
from collections import deque
from funasr import AutoModel

MODEL_DIR = "./models/iic/"
# 导入配置和工具
from config import (
    SAMPLE_RATE, FORMAT, CHANNELS, 
    VAD_CHUNK_SIZE, ASR_CHUNK_SIZE, VAD_CHUNK_DURATION_MS,
    SIMILARITY_THRESHOLD, TEMP_WAV_PATH, TEACHER_WAV_PATH,
    COMMAND_KEYWORDS_STOP, COMMAND_KEYWORDS_START
)
from speaker_manager import SpeakerManager
from utils import detect_command, check_for_commands, save_temp_wav, register_teacher_from_file

class AudioStream:
    """音频流基类，所有音频输入源应继承此类"""
    def read(self, size):
        """读取指定大小的音频数据"""
        raise NotImplementedError
    
    def close(self):
        """关闭音频流"""
        pass

class MicrophoneStream(AudioStream):
    """麦克风音频流实现"""
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=VAD_CHUNK_SIZE
        )
    
    def read(self, size):
        return self.stream.read(size)
    
    def close(self):
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p') and self.p:
            self.p.terminate()

class RecognitionState:
    """管理语音识别的状态"""
    def __init__(self, dialog_mode: bool = False):
        self.vad_cache = {}
        self.asr_cache = {}
        self.asr_buffer = bytearray()
        self.spk_buffer = []
        self.pre_buffer = deque(maxlen=3)
        self.is_speaking = False
        self.current_speaker = "[识别中]"
        self.is_speaker_identified = False
        self.current_sentence_text = ""
        self.last_asr_text = ""
        self.last_line_len = 0
        self.last_voice_time = time.time()
        self.asr_chunk_size = [0, 10, 5]
        self.encoder_chunk_look_back = 4
        self.decoder_chunk_look_back = 1
        self.dialog_mode = dialog_mode  # 是否启用“开始/停止”指令模式
        self.session_started = not dialog_mode  # 普通 ASR 直接开始
        self.pending_stop_command = None  # 记录待处理的停止命令
        self.stop_command_processed = False  # 标记是否已处理停止命令

    def reset_for_new_sentence(self):
        """重置状态以开始新句子"""
        self.asr_cache = {}
        self.asr_buffer = bytearray()
        self.spk_buffer = []
        self.current_speaker = "[识别中]"
        self.is_speaker_identified = False
        self.current_sentence_text = ""
        self.last_asr_text = ""
        self.last_line_len = 0
        self.pending_stop_command = None
        self.stop_command_processed = False

class RealtimeAssistant:
    def __init__(self):
        self.model_asr = None
        self.model_vad = None
        self.model_spk = None
        self.model_punc = None
        self.speaker_mgr = None
        self.all_results = []
        self.stop_requested = False
        self.stop_requested_by_role = None
        self.dialog_mode = False  # 运行时模式：True=对话/课堂指令模式
        self._init_models()
        self._init_speaker_manager()

    def _init_models(self):
        """初始化所有AI模型"""
        print("正在加载模型，请稍候...")
        try:
            print("正在加载语音识别模型...")
            self.model_asr = AutoModel(
                model="paraformer-zh-streaming",
                # model=MODEL_DIR + "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
                model_revision="v2.0.4",
                disable_update=True
            )
            
            print("正在加载语音检测模型...")
            self.model_vad = AutoModel(
                model="fsmn-vad",
                # model=MODEL_DIR + "speech_fsmn_vad_zh-cn-16k-common-pytorch",
                model_revision="v2.0.4",
                disable_update=True
            )
            
            print("正在加载声纹识别模型...")
            self.model_spk = AutoModel(
                model="cam++",
                # model=MODEL_DIR + "speech_campplus_sv_zh-cn_16k-common",
                model_revision="v2.0.2",
                disable_update=True
            )
            
            print("正在加载标点符号恢复模型...")
            self.model_punc = AutoModel(
                model="ct-punc",
                # model=MODEL_DIR + "punc_ct-transformer_cn-en-common-vocab471067-large",
                model_revision="v2.0.4",
                disable_update=True
            )
            print("所有模型加载完成！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise e

    def _init_speaker_manager(self):
        """初始化声纹管理器并注册老师"""
        self.speaker_mgr = SpeakerManager(threshold=SIMILARITY_THRESHOLD)

        if not self.speaker_mgr.teacher_embeddings:
            print("检测到尚未注册老师声纹。")
            if os.path.exists(TEACHER_WAV_PATH):
                print(f"发现预置音频文件: {TEACHER_WAV_PATH}")
                register_teacher_from_file(self.model_spk, self.speaker_mgr, TEACHER_WAV_PATH)
            else:
                print(f"警告: 未找到音频文件 {TEACHER_WAV_PATH}")
                print("无法注册老师声纹。所有说话人将被识别为学生。")
        else:
            print(f"已加载老师声纹: [{self.speaker_mgr.teacher_name}]")
            print(">>> 直接进入实时助手模式 <<<")

    def get_text_width(self, text):
        """计算文本的显示宽度 (中文字符计为2，其他计为1)"""
        return sum(2 if '\u4e00' <= char <= '\u9fff' else 1 for char in text)

    def _add_punctuation(self, text):
        """添加标点符号，优先使用模型，失败时使用简单后处理"""
        if not text.strip() or self.model_punc is None:
            return text
        
        try:
            result = self.model_punc.generate(text, disable_pbar=True)
            if result and len(result) > 0 and 'text' in result[0]:
                return result[0]['text']
        except Exception as e:
            print(f"标点符号恢复失败: {e}")
            traceback.print_exc()
        
        return self._simple_punctuation(text)

    def _simple_punctuation(self, text):
        """简单的标点符号后处理"""
        if not text.strip():
            return text
        
        text = text.strip()
        sentence_enders = ['吗', '呢', '吧', '啊', '呀', '啦', '哦', '哈', '嗯', '好', '是', '对', '错', '行', '可以', '不行', '知道', '明白', '理解', '同意', '反对']
        
        if any(text.endswith(ender) for ender in ['吗', '呢', '吧']):
            if not text.endswith(('？', '?')):
                text += '？'
        elif text.endswith(('。', '！', '？', '.', '!', '?')):
            return text
        elif any(punc in text for punc in ['，', '、', '：', '；', '（', '）', '“', '”', '【', '】']):
            return text
        else:
            text += '。'
        
        return text

    def _is_teacher_speaker(self, speaker):
        """检查说话人是否是老师"""
        return speaker and speaker not in ["[识别中]", "[Unknown]"] and "Teacher" in speaker

    def _get_speaker_role(self, speaker):
        if self._is_teacher_speaker(speaker):
            return "teacher"
        if not speaker or "Unknown" in speaker:
            return "unknown"
        return "student"

    def _is_authorized(self, role, command_match):
        if not command_match:
            return False
        roles = command_match.get("roles", [])
        if not roles:
            return True
        return role in roles

    def _match_command(self, text):
        if not self.dialog_mode:
            return None
        return detect_command(text)

    def _check_stop_command(self, text):
        """检查文本中是否包含停止命令"""
        if not self.dialog_mode:
            return None
        cmd = check_for_commands(text)
        stop_commands = COMMAND_KEYWORDS_STOP
        return cmd if cmd in stop_commands else None

    def _check_start_command(self, text):
        """检查文本中是否包含开始/上课命令"""
        if not self.dialog_mode:
            return None
        cmd = check_for_commands(text)
        start_commands = COMMAND_KEYWORDS_START
        if cmd in start_commands:
            return cmd
        # 简单模糊匹配：如果包含"上课"两个字
        if "上课" in text:
            return "上课"
        return None

    def _save_final_result_with_stop_command(self, speaker, text, is_teacher):
        """保存包含停止命令的结果"""
        if not text.strip():
            return
            
        punctuated_text = self._add_punctuation(text.strip())
        result = {
            'speaker': speaker,
            'text': punctuated_text,
            'raw_text': text.strip(),
            'timestamp': time.time(),
            'contains_stop_command': True,
            'triggered_by_teacher': is_teacher
        }
        
        if not is_teacher:
            result['ignored_stop_command'] = True
            
        self.all_results.append(result)
        print(f"\n✅ 保存包含停止命令的句子 ({'老师' if is_teacher else '学生'}): {speaker}: {punctuated_text}")

    def _save_final_result(self, speaker, text):
        """保存最终识别结果"""
        if not text or not text.strip():
            return None
            
        cmd_match = self._match_command(text)
        if cmd_match and cmd_match.get("type") == "stop":
            return None
            
        punctuated_text = self._add_punctuation(text.strip())
        
        result = {
            'speaker': speaker,
            'text': punctuated_text,
            'raw_text': text.strip(),
            'timestamp': time.time()
        }
        self.all_results.append(result)
        print(f"\n✅ 保存识别结果: {speaker}: {punctuated_text}")
        return result

    def _process_vad_result(self, audio_chunk_np, state):
        """处理VAD结果并更新状态"""
        try:
            res_vad = self.model_vad.generate(
                input=audio_chunk_np, 
                cache=state.vad_cache, 
                is_final=False, 
                chunk_size=VAD_CHUNK_DURATION_MS,
                disable_pbar=True
            )
            
            vad_segments = res_vad[0]['value'] if res_vad else []
            
            for segment in vad_segments:
                if segment[0] != -1:
                    # 语音开始
                    state.is_speaking = True
                    self._prepend_pre_buffer_audio(state)
                    self._print_new_line_header(state) # 传入 state 对象
                
                if segment[1] != -1:
                    # 语音结束
                    self._handle_speech_end(state)
                    
        except Exception as e:
            print(f"\nVAD处理错误: {e}")
            traceback.print_exc()

    def _prepend_pre_buffer_audio(self, state):
        """将预录制缓冲区的音频加入处理缓冲区"""
        for chunk in state.pre_buffer:
            state.asr_buffer.extend(chunk)
            state.spk_buffer.append(np.frombuffer(chunk, dtype=np.int16))

    def _print_new_line_header(self, state):
        """打印新行头"""
        if not state.session_started:
            return 0
        line_content = f"{state.current_speaker}: "
        print(f"\r{line_content}", end="", flush=True)
        return self.get_text_width(line_content)

    def _handle_sentence_completion(self, state, final_text):
        """
        处理句子完成，统一进行权限检查和保存
        Returns: bool - 是否需要停止识别
        """
        if not final_text.strip():
            return False
            
        punctuated_text = self._add_punctuation(final_text.strip())
        role = self._get_speaker_role(state.current_speaker)
        is_teacher = (role == "teacher")
        
        # === [新增逻辑] 检查是否还未开始上课 ===
        if not state.session_started:
            start_cmd = self._match_command(final_text)
            
            # 只有老师说“上课”才有效 (如果调试模式下强制Teacher，这里自然会过)
            if start_cmd and start_cmd.get("type") == "start" and self._is_authorized(role, start_cmd):
                state.session_started = True  # 标记为已开始
                
                # 保存这句话（作为第一句）
                result = {
                    'speaker': state.current_speaker,
                    'text': punctuated_text,
                    'raw_text': final_text.strip(),
                    'timestamp': time.time()
                }
                self.all_results.append(result)
                print(f"\n🔔  [{state.current_speaker}] 宣布上课，开始正式记录会议内容...")
                print(f"✅ 保存上课指令: {state.current_speaker}: {punctuated_text}")
                return False
            else:
                # 还没开始上课，忽略这句话
                # print(f"\n💤  (未上课) 忽略: {state.current_speaker}: {punctuated_text}")
                return False
        
        # === 以下是原有的逻辑（已开始上课） ===
        
        # 检查是否包含停止命令
        cmd_match = self._match_command(final_text)
        stop_command = cmd_match.get("keyword") if cmd_match else None
        
        # 构建结果对象
        result = {
            'speaker': state.current_speaker,
            'text': punctuated_text,
            'raw_text': final_text.strip(),
            'timestamp': time.time()
        }
        
        # 处理停止命令逻辑
        if cmd_match and cmd_match.get("type") == "stop":
            result['contains_stop_command'] = True
            result['triggered_by_teacher'] = is_teacher
            result['triggered_by_role'] = role
            authorized = self._is_authorized(role, cmd_match)
            result['authorized'] = authorized
            
            if authorized:
                self.stop_requested_by_role = role
                print(f"\n🛑 老师要求下课: {stop_command}")
                # 原有的 _save_final_result_with_stop_command 逻辑现在被简化为 append + return True
                self.all_results.append(result)
                print(">>> 停止识别。")
                return True
            else:
                print(f"\nℹ️  学生说 '{stop_command}'，但只有老师可以停止识别")
                self.all_results.append(result)
                return False

        # 常规保存
        self.all_results.append(result)
        print(f"\n✅ 保存识别结果: {state.current_speaker}: {punctuated_text}")
        return False


    def _handle_speech_end(self, state):
        """处理语音结束 - 重构版本"""
        state.is_speaking = False
        final_speaker = state.current_speaker
        final_text = ""
        
        try:
            if len(state.asr_buffer) > 0:
                asr_chunk_np = np.frombuffer(state.asr_buffer, dtype=np.int16)
                res_asr = self.model_asr.generate(
                    input=asr_chunk_np, 
                    cache=state.asr_cache, 
                    is_final=True,
                    chunk_size=state.asr_chunk_size,
                    encoder_chunk_look_back=state.encoder_chunk_look_back, 
                    decoder_chunk_look_back=state.decoder_chunk_look_back,
                    disable_pbar=True
                )
                
                if res_asr:
                    text = res_asr[0]['text']
                    delta = text[len(state.last_asr_text):] if text.startswith(state.last_asr_text) else text
                    final_text = state.current_sentence_text + delta
            else:
                final_text = state.current_sentence_text
                
            # 统一处理句子完成
            if final_text.strip():
                should_stop = self._handle_sentence_completion(state, final_text)
                if should_stop:
                    self.stop_requested = True
                
                # [修改] 只有在会议开始后，才打印 "句子完成"
                if state.session_started:
                    print(f"\n📝 句子完成: {final_speaker}: {final_text}")
            else:
                print()  # 没有内容，只换行
                
        except Exception as e:
            print(f"\nASR处理错误: {e}")
            traceback.print_exc()
            if state.current_sentence_text.strip():
                # 出错时也统一处理
                should_stop = self._handle_sentence_completion(state, state.current_sentence_text)
                if should_stop:
                    self.stop_requested = True
                print(f"\n📝 句子完成 (ASR错误): {final_speaker}: {state.current_sentence_text}")
        
        state.reset_for_new_sentence()


    def _process_asr_chunk(self, audio_chunk, state):
        """处理ASR块 - 移除实时停止命令检查"""
        state.asr_buffer.extend(audio_chunk)
        
        if len(state.asr_buffer) >= ASR_CHUNK_SIZE * 2:
            chunk_bytes = state.asr_buffer[:ASR_CHUNK_SIZE * 2]
            state.asr_buffer = state.asr_buffer[ASR_CHUNK_SIZE * 2:]
            
            asr_chunk_np = np.frombuffer(chunk_bytes, dtype=np.int16)
            
            try:
                res_asr = self.model_asr.generate(
                    input=asr_chunk_np, 
                    cache=state.asr_cache, 
                    is_final=False, 
                    chunk_size=state.asr_chunk_size,
                    encoder_chunk_look_back=state.encoder_chunk_look_back, 
                    decoder_chunk_look_back=state.decoder_chunk_look_back,
                    disable_pbar=True
                )
                
                if res_asr:
                    text = res_asr[0]['text']
                    if text:
                        delta = text[len(state.last_asr_text):] if text.startswith(state.last_asr_text) else text
                        state.current_sentence_text += delta
                        state.last_asr_text = text
                        self._refresh_display_line(state)
                        
                        # 移除实时停止命令检查 - 改为在句子结束时统一处理
                        
            except Exception as e:
                print(f"\nASR处理错误: {e}")
                traceback.print_exc()

    def _refresh_display_line(self, state):
        """刷新显示行"""
        if not state.session_started:
            return
        line_content = f"{state.current_speaker}: {state.current_sentence_text}"
        current_width = self.get_text_width(line_content)
        padding_len = max(0, state.last_line_len - current_width + 4)
        padding = " " * padding_len
        
        print(f"\r{line_content}{padding}", end="", flush=True)
        state.last_line_len = current_width

    def _identify_speaker(self, state):
        """识别说话人声纹"""
        # # ============== [调试代码开始] ==============
        # # 强制将所有说话人设置为 "Teacher"
        # state.current_speaker = "Teacher" 
        # state.is_speaker_identified = True
        # return # 直接返回，不执行后面真正的AI识别
        # # ============== [调试代码结束] ==============
        if state.is_speaker_identified or len(state.spk_buffer) < 6:
            return
            
        full_audio = np.concatenate(state.spk_buffer)
        save_temp_wav(full_audio, SAMPLE_RATE, TEMP_WAV_PATH)
        
        try:
            spk_res = self.model_spk.generate(TEMP_WAV_PATH, disable_pbar=True)
            if spk_res and len(spk_res) > 0 and 'spk_embedding' in spk_res[0]:
                emb = spk_res[0]['spk_embedding']
 # 修复CUDA张量转换问题：将设备上的张量移至CPU
                if hasattr(emb, 'cpu'):
                    emb = emb.cpu().numpy()
                new_speaker = self.speaker_mgr.identify(emb)
                
                if new_speaker != state.current_speaker:
                    state.current_speaker = new_speaker
                    self._refresh_display_line(state)
        except Exception as e:
            print(f"\n声纹识别错误: {e}")
            traceback.print_exc()
            state.current_speaker = "[Unknown]"
        finally:
            if os.path.exists(TEMP_WAV_PATH):
                os.remove(TEMP_WAV_PATH)
        
        state.is_speaker_identified = True
        return

    def _process_remaining_audio(self, state):
        """处理剩余音频数据 - 增强版，确保不丢失已识别文本"""
        if self.stop_requested or not state.is_speaking:
            return
            
        print("\n🔄 处理剩余音频数据...")
        
        # 保护性检查：即使ASR处理失败，也要保存已累积的文本
        try:
            if len(state.asr_buffer) > 0:
                asr_chunk_np = np.frombuffer(state.asr_buffer, dtype=np.int16)
                res_asr = self.model_asr.generate(
                    input=asr_chunk_np, 
                    cache=state.asr_cache, 
                    is_final=True,
                    chunk_size=state.asr_chunk_size,
                    encoder_chunk_look_back=state.encoder_chunk_look_back, 
                    decoder_chunk_look_back=state.decoder_chunk_look_back,
                    disable_pbar=True
                )
                
                if res_asr:
                    text = res_asr[0]['text']
                    if text.strip():
                        final_text = state.current_sentence_text + text
                        # 检查停止命令并保存
                        should_stop = self._handle_sentence_completion(state, final_text)
                        if should_stop:
                            self.stop_requested = True
                        print(f"\n📝 处理完成: {state.current_speaker}: {final_text}")
                        return  # 正常处理完成，直接返回
        except Exception as e:
            print(f"\n剩余音频处理错误: {e}")
            traceback.print_exc()
        
        # Fallback机制：如果ASR处理失败，保存已累积的文本
        if state.current_sentence_text.strip() and not self.stop_requested:
            should_stop = self._handle_sentence_completion(state, state.current_sentence_text)
            if should_stop:
                self.stop_requested = True
            print(f"\n⚠️  Fallback: 保存已累积文本 (ASR处理失败): {state.current_speaker}: {state.current_sentence_text}")
    def run_stream(self, audio_stream, timeout=30, mode="plain"):
        """
        流式处理音频输入 - 重构版本
        Args:
            audio_stream: 生成16bit pcm音频数据的生成器
            timeout: 无语音输入时的超时时间(秒)
            mode: 模式选择，"plain"=普通ASR，"dialog"=启用开始/停止指令
        Returns:
            list: 所有识别结果
        """
        dialog_mode = (mode == "dialog")
        self.dialog_mode = dialog_mode  # 保存当前会话模式（影响指令处理）

        print("\n" + "="*50)
        print("  流式语音识别模式已启动...")
        print("  等待音频数据输入...")
        if dialog_mode:
            print("  【注意】请老师先说 “上课” 或 “开始上课” 来激活记录！")
            print("  只有老师可以说'下课'或'停止记录'来结束识别")
        else:
            print("  【注意】普通 ASR 模式，无需“上课/下课”指令")
        print("="*50)
        
        # 重置状态
        self.all_results = []
        self.stop_requested = False
        self.stop_requested_by_role = None
        state = RecognitionState(dialog_mode=dialog_mode)
        
        try:
            for audio_chunk in audio_stream:
                if len(audio_chunk) == 0:
                    continue
                
                state.last_voice_time = time.time()
                audio_chunk_np = np.frombuffer(audio_chunk, dtype=np.int16)
                
                # 处理VAD
                self._process_vad_result(audio_chunk_np, state)
                
                # 处理正在说话的情况
                if state.is_speaking:
                    self._process_asr_chunk(audio_chunk, state)
                    if not state.is_speaker_identified:
                        state.spk_buffer.append(audio_chunk_np)
                        self._identify_speaker(state)
                
                # 检查停止命令
                if self.stop_requested:
                    print("\n⏹️  老师指令，结束识别...")
                    break
                
                # 更新预录制缓冲区
                state.pre_buffer.append(audio_chunk)
                
                # 检查超时
                if time.time() - state.last_voice_time > timeout and not state.is_speaking:
                    print(f"\n⏰ 超时 ({timeout}秒无输入)，停止处理...")
                    break
            
            # 处理剩余数据
            self._process_remaining_audio(state)
            
            print(f"\n✅ 识别完成，共识别到 {len(self.all_results)} 个句子")
            return self.all_results
            
        except KeyboardInterrupt:
            print("\n⏹️  用户中断识别...")
            if state.current_sentence_text.strip() and not self.stop_requested:
                self._save_final_result(state.current_speaker, state.current_sentence_text)
            return self.all_results
        except Exception as e:
            print(f"\n❌ 处理错误: {e}")
            traceback.print_exc()
            if state.current_sentence_text.strip() and not self.stop_requested:
                self._save_final_result(state.current_speaker, state.current_sentence_text)
            return self.all_results

    def run(self, mode="plain"):
        """兼容性方法，使用麦克风流"""
        return self.run_stream(MicrophoneStream(), mode=mode)

def main():
    assistant = RealtimeAssistant()
    results = assistant.run()
    if results:
        print("\n=== 所有识别结果 ===")
        for i, result in enumerate(results, 1):
            if 'contains_stop_command' in result:
                if result.get('triggered_by_teacher', False):
                    print(f"{i}. {result['speaker']}: {result['text']} (老师触发停止)")
                else:
                    print(f"{i}. {result['speaker']}: {result['text']} (学生说停止命令，已忽略)")
            else:
                print(f"{i}. {result['speaker']}: {result['text']}")

if __name__ == "__main__":
    main()
