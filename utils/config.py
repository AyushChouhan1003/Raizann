# ============================================================================
# FILE: utils/config.py - Configuration Settings
# ============================================================================
"""Optimized configuration for Raizann AI."""

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application configuration settings."""
    
    # LLM Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL_CHAT = "gpt-4o-mini"
    CHAT_LLM_TEMPERATURE = 0.7
    EMBEDDING_MODEL = "text-embedding-3-large"
    
    # Audio Configuration
    STT_MODEL = "gpt-4o-mini-transcribe"
    TTS_MODEL = "gpt-4o-mini-tts"
    TTS_VOICE = "fable"
    AUDIO_FORMAT = "mp3"
    
    # Database Configuration
    MONGO_URL = os.getenv("MONGODB_URI")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    # Performance Settings
    MAX_WORKERS = 4
    MESSAGE_BUFFER_SIZE = 100
    CACHE_SIZE = 128
    SESSION_TTL_SECONDS = 900
    LOGGING_LEVEL = "INFO"
    LLM_MODEL_DATA = "gpt-4"