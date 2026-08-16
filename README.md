# Orato Realtime Hindi & Hinglish ASR Service

A production-grade, low-latency Realtime Automatic Speech Recognition (ASR) service powered by [`tryorato/orato-asr-hindi-v1`](https://huggingface.co/tryorato/orato-asr-hindi-v1), a supervised fine-tune of `Qwen3-ASR-0.6B` optimized for conversational Hindi, Hinglish, and English speech.

---

## Features

- **Bi-directional WebSocket Streaming (`/ws/transcribe`)**: Real-time 16kHz PCM audio streaming with partial interim updates and finalized utterance events.
- **REST Transcription API (`/api/v1/transcribe`)**: Single-request transcription for `.wav`, `.mp3`, `.m4a`, `.ogg`, `.webm`, `.flac` files.
- **Hardware Acceleration**: Automatic CUDA detection (`bfloat16`/`float16` with SDPA attention) and CPU fallback (`float32`).
- **Interactive Web Interface**: Built-in dark-themed UI featuring live microphone streaming, audio waveform visualizer, VU level meter, and copy/export tools.
- **Python CLI Client (`client.py`)**: Real-time WebSocket streaming simulation or fast REST transcription directly from the terminal.
- **Hugging Face Authentication**: Automatic credential loading from `.env` (`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`).

---

## Quickstart

### 1. Environment Setup & Dependencies

Install required dependencies:
```bash
pip install -r requirements.txt
```

Verify that `.env` contains your Hugging Face access token:
```env
HF_TOKEN=hf_your_token_here
```

### 2. Start the ASR Service

Run the server on `http://127.0.0.1:8000`:
```bash
python server.py
```
Or with Uvicorn:
```bash
uvicorn server.py --host 0.0.0.0 --port 8000
```

### 3. Open Web UI

Open your browser and navigate to:
```
http://127.0.0.1:8000
```
- Click **Start Realtime Mic** to transcribe live speech from your microphone.
- Drag and drop audio files for instant transcription.

---

## API Endpoints

### 1. WebSocket Streaming Endpoint
- **URL**: `ws://127.0.0.1:8000/ws/transcribe?language=Hindi`
- **Audio Format**: 16kHz, 16-bit Mono PCM (binary bytes) or JSON frames with Base64 audio.
- **Messages**:
  - `{"type": "connected", "session_id": "...", "model": "..."}`
  - `{"type": "partial", "text": "...", "language": "Hindi", "latency_ms": 45}`
  - `{"type": "final", "text": "...", "language": "Hindi", "duration_sec": 2.1}`
  - `{"type": "speech_start"}`, `{"type": "speech_end"}`

### 2. REST Transcription
- **URL**: `POST /api/v1/transcribe`
- **Payload**: `multipart/form-data` with `file`, `language` (optional, default: "Hindi").
- **Example Response**:
```json
{
  "success": true,
  "filename": "speech.wav",
  "text": "नमस्ते, आप कैसे हैं?",
  "language": "Hindi",
  "duration_sec": 2.4,
  "latency_ms": 180.5,
  "rtf": 0.075
}
```

### 3. Health & Status
- **URL**: `GET /health`
- **Response**:
```json
{
  "status": "healthy",
  "model": "tryorato/orato-asr-hindi-v1",
  "device": "cuda",
  "dtype": "torch.bfloat16",
  "is_ready": true
}
```

---

## Python CLI Usage

Transcribe a local file via WebSocket streaming:
```bash
python client.py --file audio.wav --mode stream
```

Transcribe via REST API:
```bash
python client.py --file audio.wav --mode rest
```
