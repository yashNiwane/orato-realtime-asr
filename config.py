import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Hugging Face Auth
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

# Model Configuration
# Precedence: explicit ASR_MODEL env var > local model_weights/ dir > official Qwen3-ASR-0.6B
MODEL_NAME_OR_PATH = os.getenv("ASR_MODEL")
if not MODEL_NAME_OR_PATH:
    LOCAL_MODEL_DIR = BASE_DIR / "model_weights"
    if LOCAL_MODEL_DIR.exists() and (LOCAL_MODEL_DIR / "model.safetensors").exists():
        MODEL_NAME_OR_PATH = str(LOCAL_MODEL_DIR)
    else:
        MODEL_NAME_OR_PATH = "Qwen/Qwen3-ASR-0.6B"

FORCED_ALIGNER = os.getenv("FORCED_ALIGNER", None)
LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "Hindi")
SAMPLE_RATE = 16000

# Device settings
DEVICE = os.getenv("DEVICE", "auto")  # "auto", "cuda", "cpu"
TORCH_DTYPE = os.getenv("TORCH_DTYPE", "auto")  # "auto", "bfloat16", "float16", "float32"
QUANTIZATION = os.getenv("QUANTIZATION", "none").lower()  # "none", "int8", "8bit", "int4", "4bit"

# Inference backend: "auto" (vLLM on CUDA if installed, else transformers), "transformers", or "vllm"
ASR_BACKEND = os.getenv("ASR_BACKEND", "auto").lower()

# vLLM backend settings (requires `pip install -U qwen-asr[vllm]`)
VLLM_GPU_MEMORY_UTILIZATION = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85"))
VLLM_MAX_NEW_TOKENS = int(os.getenv("VLLM_MAX_NEW_TOKENS", "256"))
VLLM_MAX_BATCH_SIZE = int(os.getenv("VLLM_MAX_BATCH_SIZE", "32"))

# Realtime Streaming Settings (tuned for telecalling voice agents)
STREAM_CHUNK_DURATION = float(os.getenv("STREAM_CHUNK_DURATION", "0.15")) # partial decode every 150ms
STREAM_CONTEXT_DURATION = float(os.getenv("STREAM_CONTEXT_DURATION", "3.0")) # context buffer history
VAD_ENERGY_THRESHOLD = float(os.getenv("VAD_ENERGY_THRESHOLD", "0.003")) # sensitive speech detection
SILENCE_DURATION_FLUSH = float(os.getenv("SILENCE_DURATION_FLUSH", "0.4")) # finalize after 400ms silence (turn-taking)
MAX_STREAM_TOKENS = int(os.getenv("MAX_STREAM_TOKENS", "24")) # token limit for fast interim decoding

# Native incremental streaming (vLLM backend only). Falls back to windowed re-decode if disabled/unavailable.
STREAM_USE_NATIVE_VLLM = os.getenv("STREAM_USE_NATIVE_VLLM", "true").lower() in ("true", "1", "yes")
STREAM_UNFIXED_CHUNKS = int(os.getenv("STREAM_UNFIXED_CHUNKS", "2"))   # trailing chunks eligible for revision
STREAM_UNFIXED_TOKENS = int(os.getenv("STREAM_UNFIXED_TOKENS", "5"))   # trailing tokens eligible for revision
STREAM_NATIVE_CHUNK_SEC = float(os.getenv("STREAM_NATIVE_CHUNK_SEC", "2.0"))

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
GPU_WORKERS = int(os.getenv("GPU_WORKERS", "1"))  # serialized GPU inference across sessions
