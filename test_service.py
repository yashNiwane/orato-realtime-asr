"""
Integration & Verification test for Orato Realtime ASR Service.
"""
import os
import io
import time
import numpy as np
import soundfile as sf
import httpx
import asyncio
import websockets
import json

import config


def generate_sample_wav(duration_sec: float = 2.0, output_path: str = "sample_test.wav") -> str:
    """Generates a synthetic 16kHz speech-like test audio signal."""
    sr = 16000
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # 220Hz fundamental with harmonics
    audio = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.normal(0, 0.01, len(t))
    audio = audio.astype(np.float32)
    sf.write(output_path, audio, sr)
    return output_path


async def test_websocket_streaming(port: int = config.PORT):
    url = f"ws://127.0.0.1:{port}/ws/transcribe?language=Hindi"
    print(f"[*] Testing WebSocket streaming to {url}...")
    
    async with websockets.connect(url) as ws:
        msg = await ws.recv()
        data = json.loads(msg)
        print(f"[+] WebSocket Handshake OK: {data}")
        assert data.get("type") == "connected"

        # Stream 1 second of audio
        sr = 16000
        samples = (np.sin(2 * np.pi * 300 * np.linspace(0, 1, sr)) * 32767).astype(np.int16)
        raw_bytes = samples.tobytes()

        # Send in 200ms chunks
        chunk_size = int(0.2 * sr) * 2
        for i in range(0, len(raw_bytes), chunk_size):
            await ws.send(raw_bytes[i:i + chunk_size])
            await asyncio.sleep(0.05)

        # Flush
        await ws.send(json.dumps({"action": "flush"}))
        await asyncio.sleep(0.5)
        print("[+] WebSocket stream test completed successfully.")


def test_rest_endpoints(port: int = config.PORT):
    base_url = f"http://127.0.0.1:{port}"
    print(f"[*] Testing REST health endpoint at {base_url}/health...")
    
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        r = client.get("/health")
        print(f"[+] Health response ({r.status_code}): {r.json()}")
        assert r.status_code == 200

        # Transcribe test audio
        test_wav = generate_sample_wav(1.5, "test_audio.wav")
        print(f"[*] Testing /api/v1/transcribe with '{test_wav}'...")
        with open(test_wav, "rb") as f:
            r = client.post("/api/v1/transcribe", files={"file": (test_wav, f, "audio/wav")}, data={"language": "Hindi"})
            print(f"[+] Transcribe response ({r.status_code}): {r.json()}")
            assert r.status_code == 200


if __name__ == "__main__":
    generate_sample_wav(2.0, "sample_test.wav")
    print("[+] Sample test audio created: sample_test.wav")
