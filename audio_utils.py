"""
Universal audio loader and converter using PyAV (av) and soundfile.
Supports all audio/video formats: M4A, MP3, WAV, WebM, AAC, OGG, FLAC, MP4, etc.
"""
import io
from pathlib import Path
from typing import Union
import numpy as np
import soundfile as sf
import av


def load_audio(audio_source: Union[str, Path, bytes, io.BytesIO], target_sr: int = 16000) -> np.ndarray:
    """
    Decodes any audio file, byte buffer, or stream and returns a 16kHz mono float32 numpy array.
    """
    # 1. Try PyAV (handles M4A, AAC, MP3, WebM, WAV, MP4, etc.)
    try:
        if isinstance(audio_source, (str, Path)):
            container = av.open(str(audio_source))
        elif isinstance(audio_source, bytes):
            container = av.open(io.BytesIO(audio_source))
        elif isinstance(audio_source, io.BytesIO):
            container = av.open(audio_source)
        else:
            raise ValueError(f"Unsupported audio source type: {type(audio_source)}")

        resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
        chunks = []
        
        for frame in container.decode(audio=0):
            resampled_frames = resampler.resample(frame)
            if resampled_frames:
                for rf in resampled_frames:
                    chunks.append(rf.to_ndarray()[0])
                    
        container.close()

        if chunks:
            return np.concatenate(chunks).astype(np.float32)
    except Exception:
        pass

    # 2. Fallback to soundfile (WAV, FLAC, OGG)
    try:
        if isinstance(audio_source, bytes):
            bio = io.BytesIO(audio_source)
            data, sr = sf.read(bio, dtype="float32")
        elif isinstance(audio_source, (str, Path)):
            data, sr = sf.read(str(audio_source), dtype="float32")
        elif isinstance(audio_source, io.BytesIO):
            audio_source.seek(0)
            data, sr = sf.read(audio_source, dtype="float32")
        else:
            data, sr = None, None

        if data is not None:
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != target_sr:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(sr, target_sr)
                data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
            return data
    except Exception:
        pass

    # 3. Fallback to raw 16-bit PCM bytes
    if isinstance(audio_source, bytes):
        try:
            int16_data = np.frombuffer(audio_source, dtype=np.int16)
            return (int16_data.astype(np.float32) / 32768.0)
        except Exception:
            pass

    return np.zeros((0,), dtype=np.float32)


def float32_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    """Converts a float32 audio array in [-1.0, 1.0] to 16-bit PCM bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    int16_data = (clipped * 32767.0).astype(np.int16)
    return int16_data.tobytes()
