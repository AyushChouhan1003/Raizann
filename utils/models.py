# ============================================================================
# FILE: utils/models.py - Enhanced Request/Response Models
# ============================================================================
"""Pydantic models with validation."""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List

class UserRegisterRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    name: str

class StartSessionRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=100)

class StartSessionResponse(BaseModel):
    session_id: str
    user_id: str
    session_title: str
    created_at: str

class TextMessageRequest(BaseModel):
    user_id: str
    session_id: str
    message: str = Field(..., min_length=1, max_length=5000)

class AudioMessageRequest(BaseModel):
    user_id: str
    session_id: str
    audio_base64: str
    audio_format: str = "webm"
    
    @validator('audio_base64')
    def validate_base64(cls, v):
        if not v or len(v) < 100:
            raise ValueError("Invalid audio data")
        return v

class StreamResponse(BaseModel):
    type: str  # "token", "audio_chunk", "metadata", "done", "error"
    data: Optional[str] = None
    metadata: Optional[Dict] = None

class UpdateSessionTitleRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)

# ========================================
# GUEST CHAT MODELS
# ========================================

class GuestSessionResponse(BaseModel):
    guest_id: str
    created_at: str

class GuestChatRequest(BaseModel):
    guest_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1, max_length=5000)
