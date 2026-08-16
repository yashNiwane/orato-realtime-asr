import uuid
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import config
from asr_engine import ASREngine, StreamingSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ASRServer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing ASR Engine...")
    engine = ASREngine.get_instance()
    logger.info(f"ASR Service Ready! Model: {config.MODEL_NAME_OR_PATH} on {engine.device}")
    yield
    logger.info("Shutting down ASR Service...")


app = FastAPI(
    title="Orato Realtime Hindi ASR Service",
    description="Real-time and batch Speech Recognition powered by orato-asr-hindi-v1",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    engine = ASREngine.get_instance()
    return {
        "status": "healthy" if engine.is_ready else "loading",
        "model": config.MODEL_NAME_OR_PATH,
        "device": engine.device,
        "dtype": str(engine.dtype),
        "is_ready": engine.is_ready
    }


@app.get("/api/v1/model-info")
async def model_info():
    engine = ASREngine.get_instance()
    return engine.get_model_info()


@app.post("/api/v1/transcribe")
async def transcribe_file(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    context: Optional[str] = Form(""),
    return_timestamps: bool = Form(False)
):
    engine = ASREngine.get_instance()
    if not engine.is_ready:
        raise HTTPException(status_code=503, detail="ASR Model is not yet ready.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload received.")

    try:
        result = engine.transcribe_audio(
            audio_data=audio_bytes,
            language=language or config.LANGUAGE,
            context=context or "",
            return_time_stamps=return_timestamps
        )
        return {
            "success": True,
            "filename": file.filename,
            **result
        }
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/transcribe")
async def websocket_transcribe(
    websocket: WebSocket,
    language: Optional[str] = Query(config.LANGUAGE)
):
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    engine = ASREngine.get_instance()

    session = StreamingSession(
        session_id=session_id,
        engine=engine,
        language=language or config.LANGUAGE
    )

    logger.info(f"WebSocket client connected: session={session_id}, language={session.language}")
    
    await websocket.send_json({
        "type": "connected",
        "session_id": session_id,
        "model": config.MODEL_NAME_OR_PATH,
        "language": session.language,
        "sample_rate": config.SAMPLE_RATE
    })

    try:
        while True:
            message = await websocket.receive()
            
            # Binary PCM Audio Stream
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                pcm_chunk = ASREngine.bytes_to_pcm16k(raw_bytes)
                events = session.process_chunk(pcm_chunk)
                for event in events:
                    await websocket.send_json(event)

            # JSON Control Messages
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    action = payload.get("action")

                    if action == "audio" and "data" in payload:
                        import base64
                        raw_bytes = base64.b64decode(payload["data"])
                        pcm_chunk = ASREngine.bytes_to_pcm16k(raw_bytes)
                        events = session.process_chunk(pcm_chunk)
                        for event in events:
                            await websocket.send_json(event)

                    elif action == "flush":
                        events = session.flush()
                        for event in events:
                            await websocket.send_json(event)

                    elif action == "reset":
                        session = StreamingSession(
                            session_id=session_id,
                            engine=engine,
                            language=payload.get("language", session.language)
                        )
                        await websocket.send_json({
                            "type": "reset_ack",
                            "session_id": session_id
                        })

                    elif action == "set_language":
                        new_lang = payload.get("language")
                        if new_lang:
                            session.language = new_lang
                            await websocket.send_json({
                                "type": "language_updated",
                                "language": new_lang
                            })
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: session={session_id}")
    except Exception as e:
        logger.error(f"WebSocket error in session {session_id}: {e}", exc_info=True)
    finally:
        # Flush on disconnect
        try:
            flush_events = session.flush()
            for event in flush_events:
                await websocket.send_json(event)
        except Exception:
            pass


# Mount Static Files for UI
static_path = config.BASE_DIR / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/")
async def serve_index():
    index_file = config.BASE_DIR / "static" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Orato Realtime Hindi ASR API is running. Access /docs for OpenAPI specs."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=config.HOST, port=config.PORT, reload=config.DEBUG)
