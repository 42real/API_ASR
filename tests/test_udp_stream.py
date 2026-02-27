import os
import socket
import sys
import threading
import time
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure src/ is on sys.path so tests can import the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.asr_service.main import app


def _wav_to_udp_stream(wav_path: str, udp_address: str, duration_seconds: float) -> None:
    """
    参考原 audio.audio 的 _wav2stream 逻辑：
    读取 WAV -> 按 10ms 分片 -> UDP 发送。
    """
    host, port_str = udp_address.split(":")
    port = int(port_str)

    with wave.open(wav_path, "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError("WAV must be mono (1 channel)")
        if wav.getframerate() != 16000:
            raise ValueError("WAV must be 16kHz")
        if wav.getsampwidth() != 2:
            raise ValueError("WAV must be 16-bit")

        frames = wav.readframes(wav.getnframes())

    chunk_size = 320  # 10ms of 16kHz mono 16bit = 320 bytes
    total_chunks = len(frames) // chunk_size

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            for i in range(total_chunks):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size
                if end_idx > len(frames):
                    break
                chunk = frames[start_idx:end_idx]
                sock.sendto(chunk, (host, port))
                # 控制发送速率，模拟实时播放
                time.sleep(chunk_size / (16000 * 2))
                if time.time() - start_time >= duration_seconds:
                    break
    finally:
        sock.close()


def test_udp_stream_recognition():
    """
    按教程方式：HTTP 启动 -> UDP 推流 -> HTTP 停止并取结果 -> 状态检查。
    需要 ./tests/test.wav (16kHz mono 16bit WAV)。
    """
    wav_path = "./tests/test.wav"
    assert os.path.exists(wav_path), "Missing ./tests/test.wav (16kHz mono 16bit WAV)."

    udp_address = "239.168.123.161:5555"
    client = TestClient(app)

    # Start
    start_resp = client.post("/asr/start", json={})
    assert start_resp.status_code == 200

    # Status should be listening
    status_resp = client.get("/asr/status")
    assert status_resp.status_code == 200
    assert status_resp.json().get("listening") is True

    # UDP push (一次播放，不循环)
    sender = threading.Thread(target=_wav_to_udp_stream, args=(wav_path, udp_address, 5.0), daemon=True)
    sender.start()
    sender.join(timeout=10.0)
    time.sleep(1.0)

    # Stop & get result
    stop_resp = client.post("/asr/stop", json={})
    assert stop_resp.status_code == 200
    data = stop_resp.json()
    assert data.get("success") is True
    print("ASR text:\n", data.get("text", ""))

    sender.join(timeout=2.0)

    # Status should be not listening
    status_resp = client.get("/asr/status")
    assert status_resp.status_code == 200
    assert status_resp.json().get("listening") is False
