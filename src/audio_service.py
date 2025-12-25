# ============================================================================
# FILE: src/audio_service.py - Optimized Audio Processing
# ============================================================================
"""Parallel audio processing with thread pools."""

import logging
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from utils.config import Settings
import subprocess
import tempfile
import os
import asyncio

logger = logging.getLogger(__name__)

class AudioService:
    """Optimized audio service with parallel processing."""
    
    def __init__(self):
        self.client = OpenAI(api_key=Settings.OPENAI_API_KEY)
        self.executor = ThreadPoolExecutor(max_workers=Settings.MAX_WORKERS)
        logger.info("| AUDIO SERVICE                       |       DONE          |\n|-----------------------------------------------------------|")
    
    def process_base64_audio(self, audio_base64: str, audio_format: str = "webm") -> BytesIO:
        """Convert base64 audio to WAV format using ffmpeg."""
        try:
            # Remove data URL prefix
            if audio_base64.startswith("data:"):
                audio_base64 = audio_base64.split(",")[1]
            
            # Decode
            audio_bytes = base64.b64decode(audio_base64)
            
            # Save webm to temp file
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as tmp_input:
                tmp_input.write(audio_bytes)
                tmp_input_path = tmp_input.name
            
            # Convert using ffmpeg
            tmp_wav_path = tmp_input_path.replace(f'.{audio_format}', '.wav')
            
            result = subprocess.run([
                '/usr/bin/ffmpeg',  # or '/usr/bin/ffmpeg' if you need absolute path
                '-i', tmp_input_path,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y', '-loglevel', 'error',
                tmp_wav_path
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"FFmpeg conversion failed: {result.stderr}")
            
            logger.info(f"..........Audio converted from {audio_format} to WAV using ffmpeg..........🍀")
            
            # Read the converted WAV file into BytesIO
            with open(tmp_wav_path, 'rb') as f:
                wav_data = f.read()
            
            wav_io = BytesIO(wav_data)
            wav_io.seek(0)
            wav_io.name = "audio.wav"
            
            # Cleanup temp files
            os.remove(tmp_input_path)
            os.remove(tmp_wav_path)
            
            logger.info("..........Processed audio successfully with ffmpeg..........✅")
            return wav_io
        
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            # Cleanup temp files on error
            try:
                if 'tmp_input_path' in locals() and os.path.exists(tmp_input_path):
                    os.remove(tmp_input_path)
                if 'tmp_wav_path' in locals() and os.path.exists(tmp_wav_path):
                    os.remove(tmp_wav_path)
            except:
                pass
            raise ValueError(f"Invalid audio data: {str(e)}")
    
    async def transcribe_audio_async(self, audio_file: BytesIO) -> str:
        """Async transcription using thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._transcribe_sync,
            audio_file
        )
    
    def _transcribe_sync(self, audio_file: BytesIO) -> str:
        """Synchronous transcription."""
        audio_file.seek(0)
        transcript = self.client.audio.transcriptions.create(
            model=Settings.STT_MODEL,
            file=audio_file,
            response_format="text",
            temperature=0.2,
            prompt="CRITICAL: Output MUST use Devanagari script (देवनागरी) for Hindi words and Latin/Roman script (abc) for English words ONLY. Do NOT use Urdu/Arabic/Nastaliq script (اردو). Hindi = देवनागरी only. English = abc only."
        )
        logger.info(f"..........Transcribed: {transcript[:50]}.............🍀")
        return transcript.strip()
    
    async def synthesize_speech_async(self, text: str) -> bytes:
        """Async TTS using thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._synthesize_sync,
            text
        )
    
    def _synthesize_sync(self, text: str) -> bytes:
        """Synchronous TTS."""
        response = self.client.audio.speech.create(
            model=Settings.TTS_MODEL,
            voice=Settings.TTS_VOICE,
            input=text,
            response_format=Settings.AUDIO_FORMAT
        )
        logger.info(f"..........Synthesized: {text[:50]}.............🍀")
        return response.content
    
    async def synthesize_speech_streaming_async(self, text: str):
        """Stream audio chunks in real-time."""
        try:
            response = await asyncio.to_thread(
                self._synthesize_streaming_sync,
                text
            )
            
            # Yield audio chunks
            for chunk in response:
                if chunk:
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Streaming TTS error: {e}")
            raise

    def _synthesize_streaming_sync(self, text: str):
        """Synchronous streaming TTS."""
        response = self.client.audio.speech.create(
            model=Settings.TTS_MODEL,
            voice=Settings.TTS_VOICE,
            input=text,
            response_format="mp3"
        )
        
        # Stream in chunks
        for chunk in response.iter_bytes(chunk_size=1024):
            yield chunk
    
    @staticmethod
    def audio_to_base64(audio_bytes: bytes) -> str:
        """Convert audio to base64."""
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    def shutdown(self):
        """Cleanup thread pool."""
        self.executor.shutdown(wait=True)
