import uuid
import json
import logging
import asyncio
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
    logger.info(f"ASR Service Ready! Model: {config.MODEL_NAME_OR_PATH} on {engine.device} ({engine.dtype})")
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

    logger.info(f"[WS CONNECT] session={session_id}, language={session.language}, client={websocket.client}")
    
    await websocket.send_json({
        "type": "connected",
        "session_id": session_id,
        "model": config.MODEL_NAME_OR_PATH,
        "language": session.language,
        "sample_rate": config.SAMPLE_RATE,
        "device": engine.device
    })

    # Thread-safe async queue for non-blocking processing
    audio_queue = asyncio.Queue(maxsize=100)
    stop_event = asyncio.Event()

    async def sender_worker():
        """Background worker that pulls audio from queue and runs inference without blocking WebSocket I/O"""
        while not stop_event.is_set():
            try:
                pcm_chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                # Run inference in worker thread pool so event loop is completely free for ping/pong
                loop = asyncio.get_running_loop()
                events = await loop.run_in_executor(None, session.process_chunk, pcm_chunk)
                
                for event in events:
                    if websocket.client_state.name == "CONNECTED":
                        logger.info(f"[WS EVENT -> {session_id}] type={event.get('type')}, text='{event.get('text', '')}', latency={event.get('latency_ms')}ms")
                        await websocket.send_json(event)
            except Exception as e:
                logger.error(f"[WS PROCESS ERROR {session_id}]: {e}", exc_info=True)

    worker_task = asyncio.create_task(sender_worker())

    try:
        while True:
            message = await websocket.receive()
            
            if message["type"] == "websocket.disconnect":
                logger.info(f"[WS DISCONNECT MSG] session={session_id}")
                break

            # Binary PCM Audio Stream
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                pcm_chunk = ASREngine.bytes_to_pcm16k(raw_bytes)
                if not audio_queue.full():
                    audio_queue.put_nowait(pcm_chunk)
                else:
                    # Drop oldest if congested
                    try:
                        audio_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    audio_queue.put_nowait(pcm_chunk)

            # JSON Control Messages
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    action = payload.get("action")
                    logger.info(f"[WS ACTION {session_id}] action={action}")

                    if action == "ping":
                        await websocket.send_json({"type": "pong"})

                    elif action == "flush":
                        loop = asyncio.get_running_loop()
                        events = await loop.run_in_executor(None, session.flush)
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
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect as e:
        logger.info(f"[WS CLIENT CLOSED] session={session_id}, code={e.code}, reason={e.reason}")
    except Exception as e:
        logger.error(f"[WS UNHANDLED ERROR] session={session_id}: {e}", exc_info=True)
    finally:
        stop_event.set()
        worker_task.cancel()
        logger.info(f"[WS CLEANUP COMPLETE] session={session_id}")


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
