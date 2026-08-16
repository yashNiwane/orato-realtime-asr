import time
import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Union
import qwen_asr
from qwen_asr.inference.utils import parse_asr_output, normalize_language_name

import config
from audio_utils import load_audio, float32_to_pcm16_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ASREngine")


class ASREngine:
    _instance: Optional["ASREngine"] = None

    def __init__(self):
        self.device = self._select_device()
        self.dtype = self._select_dtype()
        self.model_name = config.MODEL_NAME_OR_PATH
        self.model: Optional[qwen_asr.Qwen3ASRModel] = None
        self.is_ready = False
        self._load_model()

    @classmethod
    def get_instance(cls) -> "ASREngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _select_device(self) -> str:
        if config.DEVICE != "auto":
            return config.DEVICE
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _select_dtype(self) -> torch.dtype:
        if config.TORCH_DTYPE == "bfloat16":
            return torch.bfloat16
        elif config.TORCH_DTYPE == "float16":
            return torch.float16
        elif config.TORCH_DTYPE == "float32":
            return torch.float32

        # Auto selection
        if self.device == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    def _load_model(self):
        logger.info(f"Loading ASR model '{self.model_name}' on device '{self.device}' with dtype '{self.dtype}'...")
        start_time = time.perf_counter()

        try:
            device_map = None if self.device == "cuda" else self.device
            
            # Load with HF Token
            self.model = qwen_asr.Qwen3ASRModel.from_pretrained(
                self.model_name,
                token=config.HF_TOKEN,
                dtype=self.dtype,
                device_map=device_map,
                attn_implementation="sdpa" if self.device == "cuda" else None
            )

            if self.device == "cuda":
                self.model.model = self.model.model.to("cuda")

            load_elapsed = time.perf_counter() - start_time
            self.is_ready = True
            logger.info(f"Model successfully loaded in {load_elapsed:.2f}s!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            self.is_ready = False
            raise

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": str(self.dtype),
            "is_ready": self.is_ready,
            "sample_rate": config.SAMPLE_RATE,
            "default_language": config.LANGUAGE
        }

    @staticmethod
    def bytes_to_pcm16k(audio_bytes: bytes) -> np.ndarray:
        return load_audio(audio_bytes, target_sr=config.SAMPLE_RATE)

    @staticmethod
    def calculate_rms(audio_chunk: np.ndarray) -> float:
        if len(audio_chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio_chunk))))

    @torch.no_grad()
    def fast_stream_infer(self, wav: np.ndarray, context: str = "", language: str = "Hindi", max_tokens: int = config.MAX_STREAM_TOKENS) -> Dict[str, Any]:
        """
        Fast low-latency inference specifically tuned for real-time interim streaming.
        Generates fewer tokens to achieve sub-100ms partial response times.
        """
        if not self.is_ready or self.model is None:
            return {"text": "", "language": language, "latency_ms": 0}

        t0 = time.perf_counter()
        forced_lang = normalize_language_name(language) if language else None
        prompt_text = self.model._build_text_prompt(context=context, force_language=forced_lang)

        inputs = self.model.processor(
            text=[prompt_text],
            audio=[wav],
            return_tensors="pt",
            padding=True
        )
        inputs = inputs.to(self.model.model.device).to(self.model.model.dtype)

        text_ids = self.model.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False
        )

        decoded = self.model.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        raw_out = decoded[0] if decoded else ""
        lang, txt = parse_asr_output(raw_out, user_language=forced_lang)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "text": txt.strip(),
            "language": lang or language,
            "latency_ms": round(elapsed_ms, 1)
        }

    def transcribe_audio(
        self,
        audio_data: Union[np.ndarray, str, bytes],
        language: Optional[str] = None,
        context: str = "",
        return_time_stamps: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe an audio array, file path, or audio bytes.
        """
        if not self.is_ready or self.model is None:
            raise RuntimeError("ASR Model is not initialized.")

        target_language = language or config.LANGUAGE
        
        # Prepare audio input
        if isinstance(audio_data, bytes):
            wav = load_audio(audio_data, target_sr=config.SAMPLE_RATE)
            audio_input = (wav, config.SAMPLE_RATE)
            duration_sec = len(wav) / config.SAMPLE_RATE
        elif isinstance(audio_data, np.ndarray):
            wav = audio_data.astype(np.float32)
            audio_input = (wav, config.SAMPLE_RATE)
            duration_sec = len(wav) / config.SAMPLE_RATE
        elif isinstance(audio_data, str):
            wav = load_audio(audio_data, target_sr=config.SAMPLE_RATE)
            audio_input = (wav, config.SAMPLE_RATE)
            duration_sec = len(wav) / config.SAMPLE_RATE
        else:
            raise ValueError(f"Unsupported audio type: {type(audio_data)}")

        start_time = time.perf_counter()
        
        results = self.model.transcribe(
            audio=audio_input,
            context=context,
            language=target_language,
            return_time_stamps=return_time_stamps
        )
        
        elapsed_sec = time.perf_counter() - start_time
        rtf = (elapsed_sec / duration_sec) if duration_sec > 0 else 0.0

        res = results[0] if results else None
        text = res.text.strip() if res else ""
        detected_language = res.language if res else target_language
        timestamps = res.time_stamps if res and return_time_stamps else None

        return {
            "text": text,
            "language": detected_language,
            "duration_sec": round(duration_sec, 3),
            "latency_ms": round(elapsed_sec * 1000, 2),
            "rtf": round(rtf, 3),
            "timestamps": timestamps
        }


class StreamingSession:
    """
    Stateful streaming session for real-time speech transcription.
    As speech is detected, produces rapid partial transcriptions with prefix rollback.
    """
    def __init__(
        self,
        session_id: str,
        engine: ASREngine,
        language: str = config.LANGUAGE,
        chunk_duration: float = config.STREAM_CHUNK_DURATION,
        context_duration: float = config.STREAM_CONTEXT_DURATION
    ):
        self.session_id = session_id
        self.engine = engine
        self.language = language
        self.chunk_samples = int(chunk_duration * config.SAMPLE_RATE)
        self.context_samples = int(context_duration * config.SAMPLE_RATE)
        self.silence_samples_flush = int(config.SILENCE_DURATION_FLUSH * config.SAMPLE_RATE)

        # Buffers
        self.audio_buffer = np.zeros((0,), dtype=np.float32)
        self.utterance_buffer = np.zeros((0,), dtype=np.float32)
        self.silence_counter = 0
        self.is_speaking = False
        self.confirmed_transcript = ""
        self.last_partial_transcript = ""

    def process_chunk(self, pcm_chunk: np.ndarray) -> List[Dict[str, Any]]:
        """
        Appends incoming audio chunk, manages VAD, and yields live partial & final transcripts.
        """
        events = []
        if len(pcm_chunk) == 0:
            return events

        self.audio_buffer = np.concatenate([self.audio_buffer, pcm_chunk])
        self.utterance_buffer = np.concatenate([self.utterance_buffer, pcm_chunk])

        # Voice Activity Detection (VAD) Energy calculation
        rms = self.engine.calculate_rms(pcm_chunk)
        if rms >= config.VAD_ENERGY_THRESHOLD:
            self.silence_counter = 0
            if not self.is_speaking:
                self.is_speaking = True
                events.append({
                    "type": "speech_start",
                    "session_id": self.session_id,
                    "rms": round(rms, 5)
                })
        else:
            self.silence_counter += len(pcm_chunk)

        # Utterance Completion (user paused speaking for SILENCE_DURATION_FLUSH)
        if self.is_speaking and self.silence_counter >= self.silence_samples_flush:
            if len(self.utterance_buffer) >= int(0.35 * config.SAMPLE_RATE):
                final_res = self.engine.transcribe_audio(
                    audio_data=self.utterance_buffer,
                    language=self.language,
                    context=self.confirmed_transcript[-100:] if self.confirmed_transcript else ""
                )
                text = final_res["text"].strip()
                if text and text != "<unintelligible>":
                    self.confirmed_transcript = (self.confirmed_transcript + " " + text).strip()
                    events.append({
                        "type": "final",
                        "session_id": self.session_id,
                        "text": text,
                        "cumulative_text": self.confirmed_transcript,
                        "language": final_res["language"],
                        "duration_sec": final_res["duration_sec"],
                        "latency_ms": final_res["latency_ms"]
                    })

            # Reset for next sentence
            self.utterance_buffer = np.zeros((0,), dtype=np.float32)
            self.audio_buffer = np.zeros((0,), dtype=np.float32)
            self.is_speaking = False
            self.last_partial_transcript = ""
            events.append({
                "type": "speech_end",
                "session_id": self.session_id
            })
            return events

        # Live Interim Partial Decode while speaking
        if self.is_speaking and len(self.audio_buffer) >= self.chunk_samples:
            # Decode the current sliding window
            segment = self.utterance_buffer[-self.context_samples:]
            partial_res = self.engine.fast_stream_infer(
                wav=segment,
                context=self.confirmed_transcript[-80:] if self.confirmed_transcript else "",
                language=self.language,
                max_tokens=config.MAX_STREAM_TOKENS
            )
            partial_text = partial_res["text"].strip()
            
            if partial_text and partial_text != "<unintelligible>" and partial_text != self.last_partial_transcript:
                self.last_partial_transcript = partial_text
                events.append({
                    "type": "partial",
                    "session_id": self.session_id,
                    "text": partial_text,
                    "cumulative_text": (self.confirmed_transcript + " " + partial_text).strip(),
                    "language": partial_res["language"],
                    "latency_ms": partial_res["latency_ms"]
                })
            # Reset chunk buffer
            self.audio_buffer = np.zeros((0,), dtype=np.float32)

        return events

    def flush(self) -> List[Dict[str, Any]]:
        """
        Forces flush of any remaining buffered audio to produce final transcription.
        """
        events = []
        if len(self.utterance_buffer) >= int(0.25 * config.SAMPLE_RATE):
            final_res = self.engine.transcribe_audio(
                audio_data=self.utterance_buffer,
                language=self.language,
                context=self.confirmed_transcript[-100:] if self.confirmed_transcript else ""
            )
            text = final_res["text"].strip()
            if text and text != "<unintelligible>":
                self.confirmed_transcript = (self.confirmed_transcript + " " + text).strip()
                events.append({
                    "type": "final",
                    "session_id": self.session_id,
                    "text": text,
                    "cumulative_text": self.confirmed_transcript,
                    "language": final_res["language"],
                    "duration_sec": final_res["duration_sec"],
                    "latency_ms": final_res["latency_ms"]
                })

        self.audio_buffer = np.zeros((0,), dtype=np.float32)
        self.utterance_buffer = np.zeros((0,), dtype=np.float32)
        self.is_speaking = False
        self.last_partial_transcript = ""
        return events
