import time
import logging
from typing import Dict, Any, Optional, List

import httpx

import config

logger = logging.getLogger("VoiceAgent")


class VoiceAgent:
    """
    Bridges finalized ASR transcripts to the LLM (OpenAI-compatible endpoint)
    with per-session conversation history. Emits 'agent_reply' events.
    """

    def __init__(self):
        self.enabled = config.AGENT_ENABLED
        self._histories: Dict[str, List[Dict[str, str]]] = {}

    def reset(self, session_id: str):
        self._histories.pop(session_id, None)

    def process_final(self, session_id: str, user_text: str, language: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        text = user_text.strip()
        if not text or text == "<unintelligible>":
            return None

        history = self._histories.setdefault(session_id, [])
        history.append({"role": "user", "content": text})
        if len(history) > config.AGENT_HISTORY_LIMIT:
            del history[:-config.AGENT_HISTORY_LIMIT]

        messages = [{"role": "system", "content": config.AGENT_SYSTEM_PROMPT}] + history

        t0 = time.perf_counter()
        try:
            resp = httpx.post(
                f"{config.LLM_BASE_URL}/chat/completions",
                json={
                    "model": config.LLM_MODEL,
                    "messages": messages,
                    "max_tokens": config.AGENT_MAX_TOKENS,
                    "temperature": config.AGENT_TEMPERATURE,
                },
                timeout=config.LLM_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            reply = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"[{session_id}] LLM call failed ({e}); skipping agent turn")
            return None

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        if not reply:
            return None

        history.append({"role": "assistant", "content": reply})

        return {
            "type": "agent_reply",
            "session_id": session_id,
            "text": reply,
            "user_text": text,
            "language": language,
            "latency_ms": latency_ms,
        }
