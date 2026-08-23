"""
Orato Hindi ASR Python Client
Transcribes local audio files or streams live microphone audio to the Realtime ASR WebSocket server.
"""

import sys
import os
import argparse
import asyncio
import json
import numpy as np

# Ensure Windows Console handles UTF-8 for Hindi / Devanagari characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import websockets
import httpx

import config
from audio_utils import load_audio, float32_to_pcm16_bytes


async def stream_mic(server_url: str = f"ws://127.0.0.1:{config.PORT}/ws/transcribe", language: str = "Hindi"):
    """
    Captures live microphone audio in real-time and streams to the WebSocket server.
    Words appear on the screen as you speak!
    """
    try:
        import sounddevice as sd
    except ImportError:
        print("[-] Please install sounddevice: pip install sounddevice")
        return

    url = f"{server_url}?language={language}"
    print(f"[*] Connecting to WebSocket: {url}...")

    async with websockets.connect(url) as ws:
        init_msg = await ws.recv()
        print(f"[+] Connected to Orato ASR Server!")
        print("[*] 🎙️  MICROPHONE ACTIVE: Start speaking Hindi / Hinglish now! (Press Ctrl+C to stop)")
        print("-" * 60)

        loop = asyncio.get_running_loop()
        audio_queue = asyncio.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                print(f"[!] Mic status: {status}", file=sys.stderr)
            # indata is float32 (frames, 1)
            raw_chunk = float32_to_pcm16_bytes(indata[:, 0])
            loop.call_soon_threadsafe(audio_queue.put_nowait, raw_chunk)

        # 16kHz mono, 100ms blocks = 1600 samples
        block_size = int(0.1 * config.SAMPLE_RATE)
        stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=block_size,
            callback=callback
        )

        async def send_worker():
            while True:
                chunk = await audio_queue.get()
                await ws.send(chunk)

        async def receive_worker():
            last_partial = ""
            try:
                async for msg in ws:
                    event = json.loads(msg)
                    etype = event.get("type")
                    if etype == "speech_start":
                        print("\n[Speaking...]", end="", flush=True)
                    elif etype == "partial":
                        txt = event.get("text", "")
                        if txt and txt != last_partial:
                            last_partial = txt
                            print(f"\r💬 {txt}   ", end="", flush=True)
                    elif etype == "final":
                        txt = event.get("text", "")
                        lat = event.get("latency_ms", 0)
                        print(f"\r✨ {txt}  (⏱️ {lat}ms)\n", flush=True)
                        last_partial = ""
                    elif etype == "speech_end":
                        pass
            except websockets.exceptions.ConnectionClosed:
                pass

        with stream:
            send_task = asyncio.create_task(send_worker())
            recv_task = asyncio.create_task(receive_worker())
            try:
                await asyncio.gather(send_task, recv_task)
            except (asyncio.CancelledError, KeyboardInterrupt):
                print("\n[*] Stopping microphone...")
                await ws.send(json.dumps({"action": "flush"}))
                await asyncio.sleep(0.5)


async def stream_file(file_path: str, server_url: str = f"ws://127.0.0.1:{config.PORT}/ws/transcribe", language: str = "Hindi"):
    """
    Streams any audio file (M4A, MP3, WAV, etc.) chunk-by-chunk to simulate real-time audio input.
    """
    print(f"[*] Reading audio file '{file_path}'...")
    audio_data = load_audio(file_path, target_sr=config.SAMPLE_RATE)
    
    if len(audio_data) == 0:
        print(f"[-] Error: Could not decode audio from '{file_path}'.")
        return

    duration_sec = len(audio_data) / config.SAMPLE_RATE
    raw_bytes = float32_to_pcm16_bytes(audio_data)

    url = f"{server_url}?language={language}"
    print(f"[*] Connecting to WebSocket: {url}...")

    async with websockets.connect(url) as ws:
        init_msg = await ws.recv()
        print(f"[+] Server Connected: {init_msg}")

        chunk_size = int(0.1 * config.SAMPLE_RATE) * 2  # bytes
        total_bytes = len(raw_bytes)

        async def receive_handler():
            last_partial = ""
            try:
                async for msg in ws:
                    event = json.loads(msg)
                    etype = event.get("type")
                    if etype == "partial":
                        text = event.get("text", "")
                        if text and text != last_partial:
                            last_partial = text
                            print(f"\r💬 [Live] {text}   ", end="", flush=True)
                    elif etype == "final":
                        text = event.get("text", "")
                        latency = event.get("latency_ms", 0)
                        print(f"\r✨ [Final] {text} (⏱️ {latency}ms)\n", flush=True)
                        last_partial = ""
            except websockets.exceptions.ConnectionClosed:
                pass
            except Exception as e:
                print(f"\n[!] Receive error: {e}")

        recv_task = asyncio.create_task(receive_handler())

        print(f"[*] Streaming {duration_sec:.2f}s of audio in real-time chunks...")
        for i in range(0, total_bytes, chunk_size):
            chunk = raw_bytes[i:i + chunk_size]
            await ws.send(chunk)
            await asyncio.sleep(0.08)

        await ws.send(json.dumps({"action": "flush"}))
        await asyncio.sleep(1.2)
        await ws.close()
        await recv_task


def transcribe_file_rest(file_path: str, server_url: str = f"http://127.0.0.1:{config.PORT}/api/v1/transcribe", language: str = "Hindi"):
    print(f"[*] Sending file '{file_path}' to REST endpoint {server_url}...")
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "application/octet-stream")}
        data = {"language": language}
        response = httpx.post(server_url, files=files, data=data, timeout=120.0)

    if response.status_code == 200:
        result = response.json()
        print("\n=== Transcription Result ===")
        print(f"Text:     {result.get('text')}")
        print(f"Language: {result.get('language')}")
        print(f"Duration: {result.get('duration_sec')}s")
        print(f"Latency:  {result.get('latency_ms')}ms")
        print(f"RTF:      {result.get('rtf')}")
    else:
        print(f"[-] Request failed ({response.status_code}): {response.text}")


def main():
    parser = argparse.ArgumentParser(description="Orato Realtime Hindi ASR Client")
    parser.add_argument("--mode", "-m", choices=["mic", "stream", "rest"], default="mic", help="Streaming mode: 'mic' for live microphone, 'stream' for file stream, 'rest' for batch")
    parser.add_argument("--file", "-f", type=str, help="Path to audio file (required for 'stream' and 'rest' modes)")
    parser.add_argument("--language", "-l", type=str, default="Hindi", help="Target language (default: Hindi)")
    parser.add_argument("--host", type=str, default=f"127.0.0.1:{config.PORT}", help="Server host:port")

    args = parser.parse_args()

    host = args.host.strip().rstrip("/")
    if host.startswith("https://"):
        ws_prefix = "wss://" + host[8:]
        http_prefix = host
    elif host.startswith("http://"):
        ws_prefix = "ws://" + host[7:]
        http_prefix = host
    elif host.startswith("wss://"):
        ws_prefix = host
        http_prefix = "https://" + host[6:]
    elif host.startswith("ws://"):
        ws_prefix = host
        http_prefix = "http://" + host[5:]
    elif "ngrok" in host or "trycloudflare" in host:
        ws_prefix = f"wss://{host}"
        http_prefix = f"https://{host}"
    else:
        ws_prefix = f"ws://{host}"
        http_prefix = f"http://{host}"

    if args.mode == "mic":
        ws_url = f"{ws_prefix}/ws/transcribe"
        try:
            asyncio.run(stream_mic(ws_url, args.language))
        except KeyboardInterrupt:
            print("\n[*] Exited microphone stream.")
        return

    if not args.file:
        print("[-] Please provide --file <audio_path> for stream or rest mode, or use --mode mic for live speech.")
        sys.exit(1)

    if not os.path.exists(args.file):
        print(f"[-] Error: File '{args.file}' not found.")
        sys.exit(1)

    if args.mode == "stream":
        ws_url = f"{ws_prefix}/ws/transcribe"
        asyncio.run(stream_file(args.file, ws_url, args.language))
    else:
        rest_url = f"{http_prefix}/api/v1/transcribe"
        transcribe_file_rest(args.file, rest_url, args.language)


if __name__ == "__main__":
    main()
