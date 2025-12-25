# ============================================================================
# FILE: main.py - RaizannAI API
# ============================================================================
"""Refactored API for RaizannAI."""

import logging
import warnings
from pathlib import Path


import asyncio
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from utils.config import Settings
from utils.models import (
    UserRegisterRequest, StartSessionRequest, StartSessionResponse,
    TextMessageRequest, AudioMessageRequest, UpdateSessionTitleRequest,
    GuestSessionResponse, GuestChatRequest
)
from src.storage import StorageManager
from src.llm_service import LLMService
from src.audio_service import AudioService
from src.message_processor import MessageProcessor
from src.guest_storage import GuestStorageManager
from src.emotion_service import VoiceEmotionDetector

# Ensure logs directory exists (auto-create if deleted)
Path('logs').mkdir(exist_ok=True)

# Setup logging - outputs to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/raizann.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP request logs from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

# Global services
storage: StorageManager = None
llm_service: LLMService = None
audio_service: AudioService = None
message_processor: MessageProcessor = None
guest_storage: GuestStorageManager = None
emotion_detector: VoiceEmotionDetector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global storage, llm_service, audio_service, message_processor, guest_storage, emotion_detector
    
    logger.info(" ___________________________________________________________\n|                                                           |\n|            RAIZANN AI BACKEND INITIALIZATION              |\n|___________________________________________________________|")
    logger.info(" ___________________________________________________________\n|                                                           |\n|                   INITIATING SERVICES                     |\n|___________________________________________________________|")

    storage = StorageManager()
    llm_service = LLMService()
    message_processor = MessageProcessor(storage, llm_service)
    guest_storage = GuestStorageManager(max_messages_per_guest=30)  # 15 user + 15 assistant
    audio_service = AudioService()
    
    # Initialize emotion detector
    logger.info(" ___________________________________________________________\n|                                                           |\n|                     EMOTION SERVICES                      |\n|___________________________________________________________|")
    emotion_detector = VoiceEmotionDetector(production_model_path="utils/production_model.pt")
    
    # Start guest cleanup scheduler
    cleanup_task = asyncio.create_task(guest_cleanup_scheduler())
    
    logger.info(" ___________________________________________________________\n|                                                           |\n|                 RAIZANN AI BACKEND READY                  |\n|___________________________________________________________|")
    
    yield
    
    logger.info(" ___________________________________________________________\n|                                                           |\n|             RAIZANN AI BACKEND SHUTTING DOWN              |\n|___________________________________________________________|")
    cleanup_task.cancel()
    audio_service.shutdown()
    storage.close()
    
    #  Cleanup emotion detector
    if emotion_detector:
        emotion_detector.clear_cache()
        logger.info("| EMOTION SERVICES CLEANUP            |       DONE          |\n|-----------------------------------------------------------|")
    logger.info(" ___________________________________________________________\n|                                                           |\n|              RAIZANN AI BACKEND DEACTIVATED               |\n|___________________________________________________________|")



async def guest_cleanup_scheduler():
    """Background task to cleanup inactive guest sessions every 10 minutes."""
    while True:
        try:
            await asyncio.sleep(600)  # 10 minutes
            logger.info("Running guest session cleanup...")
            guest_storage.cleanup_inactive_sessions(inactivity_threshold_minutes=30)
            logger.info(f"Active guest sessions: {guest_storage.get_session_count()}")
        except asyncio.CancelledError:
            logger.info("| GUEST CLEANUP SCHEDULER SHUTDOWN    |       DONE          |\n|-----------------------------------------------------------|")

            break
        except Exception as e:
            logger.error(f"Error in guest cleanup scheduler: {e}")

app = FastAPI(
    title="RaizannAI",
    lifespan=lifespan,
    root_path="/ai"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# USER & SESSION API
# ========================================

@app.post("/api/register")
async def register_user(request: UserRegisterRequest):
    """Register or update user demographics."""
    try:
        user = storage.register_user(request.dict())
        return {"status": "success", "user": user}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/session/start", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest):
    """Start a new chat session."""
    session_id = f"session_{uuid.uuid4().hex}"
    storage.create_session(request.user_id, session_id)
    return StartSessionResponse(
        session_id=session_id,
        user_id=request.user_id,
        session_title="New chat",
        created_at=str(asyncio.get_event_loop().time())
    )

@app.get("/api/sessions/{user_id}")
async def get_sessions(user_id: str):
    """Get user sessions."""
    return {"sessions": storage.get_user_sessions(user_id)}

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session and its messages."""
    try:
        result = storage.delete_session(session_id)
        if result:
            return {"status": "success", "message": "Session deleted"}
        else:
            return {"status": "success", "message": "Session already deleted or not found"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_session endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/api/sessions/{session_id}/title")
async def update_session_title(session_id: str, request: UpdateSessionTitleRequest):
    """Update session title."""
    try:
        result = storage.update_session_title(session_id, request.title)
        if result:
            # Fetch updated session to return complete information
            updated_session = storage.get_session(session_id)
            return {
                "status": "success", 
                "message": "Session title updated",
                "session": {
                    "session_id": updated_session.get("session_id"),
                    "session_title": updated_session.get("session_title"),
                    "user_id": updated_session.get("user_id"),
                    "last_updated": updated_session.get("last_updated").isoformat() if updated_session.get("last_updated") else None
                }
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session title: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/user/{user_id}")
async def delete_user(user_id: str):
    """Soft delete a user and all their associated data."""
    try:
        result = storage.soft_delete_user(user_id)
        
        if result["status"] == "error":
            if "not found" in result["message"]:
                raise HTTPException(status_code=404, detail=result["message"])
            elif "already deleted" in result["message"]:
                raise HTTPException(status_code=409, detail=result["message"])
            else:
                raise HTTPException(status_code=500, detail=result["message"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_user endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    """Get session history."""
    return {"messages": storage.get_recent_messages(session_id, limit=50)}


# ========================================
# CHAT API
# ========================================

@app.post("/api/chat/text/stream")
async def chat_text_stream(request: TextMessageRequest):
    """Streaming text chat with Server-Sent Events (SSE) format."""
    import json
    
    logger.info(f"..........Received text chat request for user: {request.user_id}..........🍀")
    
    async def generate():
        async for token in message_processor.process_message_text(
            request.user_id, request.session_id, request.message
        ):
            # Format as SSE with JSON
            sse_data = f"data: {json.dumps({'token': token})}\n\n"
            yield sse_data
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/api/chat/audio")
async def chat_audio(request: AudioMessageRequest):
    """Audio input -> Streaming audio output in SSE format."""
    try:
        # Store Raw Audio (Background)
        import base64
        import json
        try:
            raw_audio = base64.b64decode(request.audio_base64)
            asyncio.create_task(asyncio.to_thread(
                storage.store_audio,
                request.user_id,
                request.session_id,
                raw_audio,
                request.audio_format
            ))
        except Exception as e:
            logger.error(f"Failed to store audio: {e}")

        # Process Audio
        audio_file = await asyncio.to_thread(
            audio_service.process_base64_audio,
            request.audio_base64,
            request.audio_format
        )

        logger.info(f"..........Received audio chat request for user: {request.user_id}..........🍀")
        
        # Transcribe
        transcription = await audio_service.transcribe_audio_async(audio_file)
        if not transcription:
            raise HTTPException(status_code=400, detail="No speech detected")
        
        # --------------Detecting Audio Emotion -------------
        try:
            emotion_result = emotion_detector.detect_emotion(
                request.audio_base64,
                request.audio_format
            )
            audio_emotion = emotion_result['emotion']  # "Positive", "Neutral", or "Negative"
            logger.info(f"..........Detected emotion: {audio_emotion} (confidence: {emotion_result.get('confidence', 'N/A')})..........🍀")
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            audio_emotion = "Neutral"  # Fallback to neutral if detection fails
        
        # Get LLM Response
        full_response = ""
        async for token in message_processor.process_message_tts(
            request.user_id, request.session_id, transcription, audio_emotion
        ):
            full_response += token
            
        # Stream Audio Response in SSE format
        async def generate_audio():
            async for chunk in audio_service.synthesize_speech_streaming_async(full_response):
                # Encode audio chunk to base64 for JSON
                chunk_b64 = base64.b64encode(chunk).decode('utf-8')
                sse_data = f"data: {json.dumps({'chunk': chunk_b64})}\n\n"
                yield sse_data
        
        logger.info(f"..........Generating audio streaming response for user: {request.user_id}..........🍀")
        
        return StreamingResponse(generate_audio(), media_type="text/event-stream")
        
    except Exception as e:
        logger.error(f"Audio chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# GUEST CHAT API
# ========================================

@app.post("/api/guest/session/start", response_model=GuestSessionResponse)
async def start_guest_session():
    """Start a new guest chat session."""
    try:
        guest_id = guest_storage.create_guest_session()
        return GuestSessionResponse(
            guest_id=guest_id,
            created_at=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Error creating guest session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/guest/chat")
async def guest_chat_stream(request: GuestChatRequest):
    """Streaming guest chat with Server-Sent Events (SSE) format."""
    import json
    from datetime import datetime
    
    # Verify guest session exists
    if not guest_storage.session_exists(request.guest_id):
        raise HTTPException(status_code=404, detail="Guest session not found")
    
    # Check if guest has exceeded message limit (15 user queries)
    if not guest_storage.can_send_message(request.guest_id, max_queries=15):
        raise HTTPException(
            status_code=429,
            detail="Guest message limit reached. Maximum 15 queries allowed in guest mode. Please register for unlimited access."
        )
    
    logger.info(f"Guest chat request: {request.guest_id}")
    
    async def generate():
        # Get guest message history
        history = guest_storage.get_guest_messages(request.guest_id, limit=10)
        
        # Stream LLM response (no assessment, no profiling, no memory)
        full_response = ""
        async for token in llm_service.stream_chat_text(
            request.message,
            history,
            {},  # Empty user profile
            "",  # No memory context
            assessment_active=False
        ):
            full_response += token
            sse_data = f"data: {json.dumps({'token': token})}\n\n"
            yield sse_data
        
        # Save messages to guest storage
        guest_storage.save_guest_message(request.guest_id, "user", request.message)
        guest_storage.save_guest_message(request.guest_id, "assistant", full_response)
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.delete("/api/guest/session/{guest_id}")
async def terminate_guest_session(guest_id: str):
    """Terminate a guest session and erase all data."""
    try:
        result = guest_storage.delete_guest_session(guest_id)
        if result:
            return {"status": "success", "message": "Guest session terminated"}
        else:
            raise HTTPException(status_code=404, detail="Guest session not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error terminating guest session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================================
# Report API
# ========================================

@app.get("/api/report/{user_id}")
async def generate_user_report(user_id: str):
    """Generate comprehensive mental health report for a user."""
    from src.report_service import generate_report, render_report_html
    from fastapi.responses import HTMLResponse
    
    try:
        # Generate report (returns JSON)
        report = await generate_report(user_id, storage, llm_service)
        
        # Check if it's a status message (not a full report)
        if "status" in report and report["status"] in ["no_assessment", "assessment_in_progress", "error", "user_not_found", "insufficient_data"]:
            # Return JSON for status messages
            return report
        
        # Render full report as HTML
        html = render_report_html(report)
        return HTMLResponse(content=html)
        
    except Exception as e:
        logger.error(f"Error in report endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))




# ========================================
# Application Health API
# ========================================


@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)