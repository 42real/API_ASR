import logging
import socket
import struct
import threading
import time
from typing import Iterable, List

import pyaudio

from .asr_core.config import VAD_CHUNK_SIZE
from .speaker_audio import SpeakerAudio

logger = logging.getLogger(__name__)


def microphone_audio_stream(stop_event: threading.Event, chunk_size: int = VAD_CHUNK_SIZE) -> Iterable[bytes]:
    """
    生成麦克风音频流（16kHz、单声道、16bit PCM）。

    通过 stop_event 控制结束，确保可以被 /asr/stop 主动中止。
    """
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=chunk_size,
    )

    try:
        while not stop_event.is_set():
            data = stream.read(chunk_size, exception_on_overflow=False)
            if data:
                yield data
    finally:
        try:
            stream.stop_stream()
            stream.close()
        finally:
            pa.terminate()


class AsrSessionManager:
    """
    ASR 会话管理：保证同时只有一个监听会话。
    """

    def __init__(self, timeout_seconds: int = 30):
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._results: List[dict] = []
        self._error: Exception | None = None
        self._listening = False

    def start(self) -> None:
        with self._lock:
            if self._listening:
                raise RuntimeError("ASR session already active")

            self._stop_event = threading.Event()
            self._results = []
            self._error = None
            self._listening = True

            def _worker():
                try:
                    audio = SpeakerAudio()

                    # === ASR 入口选择（仅保留一个启用，其余注释） ===
                    # 1) 机器人方法：UDP 流（返回字符串）
                    # recognized_text = stream2text_udp(audio, "239.168.123.161:5555", duration=None, mode="dialog")
                    # self._results = [{"speaker": "Robot", "text": recognized_text}]
                    #
                    # 2) 调试1：本地麦克风输入（推荐/当前启用）
                    stream = microphone_audio_stream(self._stop_event, VAD_CHUNK_SIZE)
                    self._results = audio.process_audio_stream(stream, mode="plain")
                    #
                    # 3) 调试2：手动输入测试
                    # recognized_text = input("请输入测试文本: ")
                    # self._results = [{"speaker": "Manual", "text": recognized_text}]
                except Exception as e:
                    self._error = e
                    logger.exception("ASR worker failed")
                finally:
                    with self._lock:
                        self._listening = False

            self._thread = threading.Thread(target=_worker, daemon=True)
            self._thread.start()

    def stop(self) -> List[dict]:
        with self._lock:
            if not self._listening or self._thread is None or self._stop_event is None:
                raise RuntimeError("ASR session not active")
            self._stop_event.set()
            thread = self._thread

        thread.join(timeout=self._timeout_seconds)
        if thread.is_alive():
            raise TimeoutError("ASR stop timed out")

        if self._error is not None:
            raise self._error

        return self._results

    def status(self) -> bool:
        with self._lock:
            return self._listening


def results_to_text(results: List[dict]) -> str:
    """
    将识别结果列表拼接为文本（对齐 conv.py 的输出格式）。
    """
    if not results:
        return ""
    lines = []
    for r in results:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        speaker = r.get("speaker") or ""
        if speaker:
            lines.append(f"{speaker}: {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def stream2text_udp(
    audio: SpeakerAudio,
    udp_address: str = "239.168.123.161:5555",
    duration: float | None = 5.0,
    mode: str = "plain",
) -> str:
    """
    从 UDP 音频流识别文本（复用 SpeakerAudio 的 ASR 逻辑）。
    参考 robot-dialogue/audio/audio.py 的 stream2text 实现。
    """
    host, port_str = udp_address.split(":")
    port = int(port_str)

    # 判断是否为组播地址
    is_multicast = False
    try:
        ip_parts = list(map(int, host.split(".")))
        if len(ip_parts) == 4 and 224 <= ip_parts[0] <= 239:
            is_multicast = True
    except Exception:
        pass

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp_socket.settimeout(1.0)

    try:
        if is_multicast:
            udp_socket.bind(("", port))
            group = socket.inet_aton(host)
            mreq = struct.pack("4sL", group, socket.INADDR_ANY)
            udp_socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        else:
            udp_socket.bind((host, port))
    except Exception as e:
        udp_socket.close()
        raise Exception(f"UDP 绑定失败: {e}")

    chunk_size_bytes = 6400  # 200ms * 16000 * 2

    def udp_audio_stream_generator():
        start_time = time.time()
        buffer = b""
        try:
            while duration is None or (time.time() - start_time) < duration:
                try:
                    data, _ = udp_socket.recvfrom(4096)
                    buffer += data
                    while len(buffer) >= chunk_size_bytes:
                        chunk = buffer[:chunk_size_bytes]
                        buffer = buffer[chunk_size_bytes:]
                        yield chunk
                except socket.timeout:
                    if len(buffer) >= chunk_size_bytes:
                        chunk = buffer[:chunk_size_bytes]
                        buffer = buffer[chunk_size_bytes:]
                        yield chunk
                    continue
                except OSError:
                    break
        finally:
            if is_multicast:
                try:
                    group = socket.inet_aton(host)
                    mreq = struct.pack("4sL", group, socket.INADDR_ANY)
                    udp_socket.setsockopt(socket.IPPROTO_IP, socket.IP_DROP_MEMBERSHIP, mreq)
                except Exception:
                    pass
            udp_socket.close()

    results = audio.process_audio_stream(udp_audio_stream_generator(), mode=mode)
    lines = [f"{r.get('speaker', '')}: {r.get('text', '')}" for r in results if r.get("text")]
    return "\n".join([line for line in lines if line.strip()])
