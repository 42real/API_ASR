import os
import time
import wave
import pyaudio
import numpy as np
import scipy.io.wavfile as wavfile
from config import (
    SAMPLE_RATE, CHUNK_SIZE, FORMAT, CHANNELS, 
    TEMP_WAV_PATH, COMMAND_KEYWORDS, COMMAND_DEFINITIONS
)

def record_voice_fingerprint(model, speaker_manager):
    """录制并注册老师声纹"""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    print("\n" + "="*40)
    print("  开始注册老师声纹")
    print("  请在 3 秒后说一段话（约5秒）...")
    print("="*40)
    time.sleep(3)
    print(">>> 开始录音... <<<")

    frames = []
    for _ in range(0, int(SAMPLE_RATE / CHUNK_SIZE * 5)):  # 录制5秒
        data = stream.read(CHUNK_SIZE)
        frames.append(data)

    print(">>> 录音结束 <<<")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 保存临时文件供模型读取
    wf = wave.open(TEMP_WAV_PATH, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # 提取特征
    try:
        # 使用文件路径而不是numpy数组，避免维度问题
        # 注意：这里添加 disable_pbar=True 保持清爽
        res = model.generate(input=os.path.abspath(TEMP_WAV_PATH), disable_pbar=True)
        
        # 修正：从返回结果中正确提取 embedding
        if res and len(res) > 0 and 'spk_embedding' in res[0]:
            embedding = res[0]['spk_embedding']
            speaker_manager.save_teacher("Teacher", embedding)
            print("注册成功！")
        else:
            # 尝试直接使用返回结果（兼容旧版本或不同返回格式）
            if res is not None:
                 speaker_manager.save_teacher("Teacher", res)
                 print("注册成功！(直接保存)")
            else:
                print("注册失败：未能提取到有效声纹特征。")

    except Exception as e:
        print(f"注册失败: {e}")
    
    # 清理临时文件
    if os.path.exists(TEMP_WAV_PATH):
        os.remove(TEMP_WAV_PATH)

def register_teacher_from_file(model, speaker_manager, file_path):
    """从文件注册老师声纹 (多粒度切片版 - 增强版)"""
    if not os.path.exists(file_path):
        print(f"未找到老师录音文件: {file_path}")
        return

    print(f"正在处理老师录音文件: {file_path} ...")
    try:
        # 读取音频
        sr, audio_data = wavfile.read(file_path)
        
        # --- 增强处理开始 ---
        # 1. 检查并转换声道
        if audio_data.ndim > 1:
            print(f"检测到多声道音频 ({audio_data.shape})，正在转换为单声道...")
            audio_data = np.mean(audio_data, axis=1).astype(np.int16)
        
        # 2. 检查采样率
        if sr != SAMPLE_RATE:
            print(f"警告: 采样率不匹配 (文件:{sr} vs 系统:{SAMPLE_RATE})，建议重采样。")
        
        duration = len(audio_data) / sr
        print(f"音频时长: {duration:.2f} 秒, 样本数: {len(audio_data)}")
        # --- 增强处理结束 ---

        # 策略：将音频切成多个片段
        segment_len = 3 * SAMPLE_RATE # 3秒
        step = int(1.5 * SAMPLE_RATE) # 1.5秒步长
        
        embeddings = []
        
        # 1. 首先提取全量音频的特征 (作为基准)
        print("正在提取全量音频特征...")
        try:
            res_global = model.generate(input=os.path.abspath(file_path), disable_pbar=True)
            if res_global and 'spk_embedding' in res_global[0]:
                embeddings.append(res_global[0]['spk_embedding'])
                print("  - 全量特征提取成功")
        except Exception as e:
            print(f"  - 全量特征提取失败: {e}")

        # 2. 如果音频足够长，进行切片提取
        if len(audio_data) >= segment_len:
            # 滑动窗口切片
            count = 0
            for start in range(0, len(audio_data) - segment_len + 1, step):
                end = start + segment_len
                chunk = audio_data[start:end]
                
                # 保存临时切片
                temp_slice = f"temp_register_slice_{count}.wav" # 避免文件名冲突
                wavfile.write(temp_slice, sr, chunk) # 使用读取到的sr
                
                # 提取特征
                try:
                    res = model.generate(input=os.path.abspath(temp_slice), disable_pbar=True)
                    if res and len(res) > 0 and 'spk_embedding' in res[0]:
                        embeddings.append(res[0]['spk_embedding'])
                    else:
                        print(f"  - 提取片段 {count}: 失败 (模型未返回特征)")
                except Exception as e:
                    print(f"  - 提取片段 {count} 出错: {e}")
                finally:
                    if os.path.exists(temp_slice):
                        os.remove(temp_slice)
                
                count += 1
                    
        if len(embeddings) > 0:
            speaker_manager.save_teacher("Teacher", embeddings)
            print(f"注册成功！共提取了 {len(embeddings)} 个特征向量。")
        else:
            print("注册失败：未能提取到有效特征。")

    except Exception as e:
        print(f"注册过程出错: {e}")
        import traceback
        traceback.print_exc()

def _normalize_text(text):
    if text is None:
        return ""
    return str(text).strip()

def detect_command(text):
    """
    Detect command and return a structured match dict:
    {
        "id": "...",
        "type": "...",
        "roles": [...],
        "keyword": "...",
        "source": "exact" | "fuzzy"
    }
    """
    text_norm = _normalize_text(text)
    if not text_norm:
        return None

    # Exact match
    for cmd in COMMAND_DEFINITIONS:
        if not cmd or "id" not in cmd:
            continue
        for kw in cmd.get("keywords", []):
            if kw and kw in text_norm:
                return {
                    "id": cmd.get("id"),
                    "type": cmd.get("type"),
                    "roles": cmd.get("roles", []),
                    "keyword": kw,
                    "source": "exact",
                }

    # Optional fuzzy aliases
    for cmd in COMMAND_DEFINITIONS:
        if not cmd or "id" not in cmd:
            continue
        for kw in cmd.get("fuzzy_keywords", []):
            if kw and kw in text_norm:
                return {
                    "id": cmd.get("id"),
                    "type": cmd.get("type"),
                    "roles": cmd.get("roles", []),
                    "keyword": kw,
                    "source": "fuzzy",
                }

    return None

def check_for_commands(text):
    """
    检查文本中是否包含指令
    支持精确匹配和简单的模糊匹配
    """
    if not text:
        return None

    # 1. 精确匹配 (遍历配置的关键词)
    for cmd in COMMAND_KEYWORDS:
        if cmd in text:
            return cmd

    # 2. 模糊匹配 (针对 ASR 可能出现的同音字或断句问题)
    # 例如: "下 课" 或 "下客"

    # # 针对 "下课" 的特殊模糊处理
    # if "下" in text and ("课" in text or "客" in text):
    #     # 检查两个字是否相邻或很近 (距离小于3个字符)
    #     idx_xia = text.find("下")
    #     idx_ke = text.find("课")
    #     if idx_ke == -1:
    #         idx_ke = text.find("客")

    #     if idx_ke > idx_xia and (idx_ke - idx_xia) <= 3:
    #         return "下课 (模糊匹配)"

    # # 针对 "停止" 的特殊模糊处理
    # if "停" in text and ("止" in text or "指" in text):
    #     return "停止 (模糊匹配)"

    return None

def save_temp_wav(audio_data, sample_rate, path):
    """将numpy音频数据保存为wav文件"""
    # 确保数据是 int16 格式
    if audio_data.dtype != np.int16:
        audio_data = (audio_data * 32767).astype(np.int16)
    wavfile.write(path, sample_rate, audio_data)
