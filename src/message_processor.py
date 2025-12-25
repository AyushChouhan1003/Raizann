# ============================================================================
# FILE: src/message_processor.py - Simplified Message Processing
# ============================================================================
"""Simplified message pipeline."""

import logging
import asyncio
import json
import re
from typing import AsyncGenerator
from src.storage import StorageManager
from src.llm_service import LLMService
from utils.prompts import USER_PROFILING_PROMPT

logger = logging.getLogger(__name__)

class MessageProcessor:
    """Handles message flow."""
    
    def __init__(self, storage: StorageManager, llm_service: LLMService):
        self.storage = storage
        self.llm_service = llm_service
    
    async def process_message_tts(self, user_id: str, session_id: str, message: str, audio_emotion: str) -> AsyncGenerator[str, None]:
        """Process message and stream response."""
        try:
            # 1. Get User Profile
            user_profile = self.storage.get_user(user_id) or {}
            
            # 2. Check assessment status
            assessment_active = self.storage.get_active_assessment_status(user_id)
            
            # 3. Get Context (Parallel: Recent History + Vector Search)
            history_task = asyncio.to_thread(
                self.storage.get_recent_messages, 
                session_id, 
                limit=10
            )
            
            memory_task = self.storage.search_memory(user_id, message, mode="semantic")
            
            history, memory_context = await asyncio.gather(history_task, memory_task)
            
            # 4. Stream LLM Response with conditional assessment prompt
            full_response = ""
            async for token in self.llm_service.stream_chat_tts(
                message, history, user_profile, memory_context, audio_emotion , assessment_active
            ):
                full_response += token
                yield token
            
            # 5. Save & Vectorize (Background)
            asyncio.create_task(self._finalize(
                user_id, session_id, message, full_response
            ))
            
        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            yield "<response emotion='neutral'>Error processing request.</response>"


    async def process_message_text(self, user_id: str, session_id: str, message: str) -> AsyncGenerator[str, None]:
        """Process message and stream response."""
        try:
            # 1. Get User Profile
            user_profile = self.storage.get_user(user_id) or {}
            
            # 2. Check if we should trigger new assessment (after 7-day cooldown)
            if self.storage.should_trigger_assessment(user_id):
                self.storage.start_new_assessment_cycle(user_id)
            
            # 3. Check assessment status
            assessment_active = self.storage.get_active_assessment_status(user_id)
            
            # 3. Get Context (Parallel: Recent History + Vector Search)
            history_task = asyncio.to_thread(
                self.storage.get_recent_messages, 
                session_id, 
                limit=10
            )
            
            memory_task = self.storage.search_memory(user_id, message, mode="semantic")
            
            history, memory_context = await asyncio.gather(history_task, memory_task)
            
            # 4. Stream LLM Response with conditional assessment prompt
            full_response = ""
            async for token in self.llm_service.stream_chat_text(
                message, history, user_profile, memory_context, assessment_active
            ):
                full_response += token
                yield token
            
            # 5. Save & Vectorize (Background)
            asyncio.create_task(self._finalize(
                user_id, session_id, message, full_response
            ))
            
        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            yield "Error processing request."

    async def _finalize(self, user_id: str, session_id: str, user_msg: str, bot_msg: str):
        """Save to DB and Vector Store."""
        try:

            bot_msg = re.sub(r'<[^>]+>', '', bot_msg).strip()

            # Save raw messages
            self.storage.save_message(session_id, user_id, "user", user_msg)
            self.storage.save_message(session_id, user_id, "assistant", bot_msg)
            
            # Vectorize pair
            await self.storage.vectorize_and_store(user_id, user_msg, bot_msg)
            
            # NEW: Assessment tracking
            if self.storage.is_assessment_active(user_id):
                # Increment user message count
                msg_count = await asyncio.to_thread(
                    self.storage.increment_assessment_message_count,
                    user_id
                )
                
                # Check every 10 messages
                if msg_count % 10 == 0:
                    logger.info(f"..........10-message checkpoint for {user_id}, checking assessment progress..........🍀")
                    asyncio.create_task(self._check_assessment_progress(user_id, user_msg, bot_msg))
            
            # Check if profiling should run (every 30 messages)
            message_count = await asyncio.to_thread(
                self.storage.count_user_messages,
                user_id
            )
            
            if message_count > 0 and message_count % 30 == 0:
                logger.info(f"..........Triggering user profiling for {user_id} at {message_count} messages..........🍀")
                asyncio.create_task(self.run_user_profiling(user_id))
            
        except Exception as e:
            logger.error(f"Finalization error: {e}")
    
    async def _check_assessment_progress(self, user_id: str, recent_user_msg: str, recent_assistant_msg: str):
        """Check if assessment questions have been answered, runs every 10 messages."""
        try:
            # Get last 20 messages (10 exchanges) for context
            recent_history = await asyncio.to_thread(
                self.storage.get_user_recent_conversations,
                user_id,
                20
            )
            
            # Get current question status
            questions_status = await asyncio.to_thread(
                self.storage.get_questions_status,
                user_id
            )
            
            # Use LLM to detect which dimensions were covered
            answered_dimensions = await self.llm_service.detect_answered_dimensions(
                recent_history,
                questions_status
            )
            
            # Mark questions as answered
            for dimension in answered_dimensions:
                await asyncio.to_thread(
                    self.storage.mark_question_answered,
                    user_id,
                    dimension
                )
                logger.info(f"..........Dimension '{dimension}' answered for user {user_id}..........✅")
            
            # Check if all 8 completed (auto-checked in get_active_assessment_status)
            if answered_dimensions:
                all_complete = await asyncio.to_thread(
                    self.storage.check_all_questions_answered,
                    user_id
                )
                if all_complete:
                    logger.info(f"..........All 8 questions answered! Completing assessment for {user_id}..........✅")
                    await asyncio.to_thread(
                        self.storage.complete_assessment,
                        user_id
                    )
        except Exception as e:
            logger.error(f"Error checking assessment progress: {e}")
    
    async def run_user_profiling(self, user_id: str):
        """Extract user profile from chat history using LLM."""
        try:
            # Get chat history
            chat_history = await asyncio.to_thread(
                self.storage.get_chat_history_for_analysis,
                user_id,
                limit=30
            )
            
            if not chat_history:
                logger.warning(f"No chat history for profiling: {user_id}")
                return
            
            # Prepare prompt
            profiling_prompt = USER_PROFILING_PROMPT.replace("{{CHAT_HISTORY}}", chat_history)
            
            # Get LLM response
            json_response = await self.llm_service.get_json_response(
                profiling_prompt,
                ""  # The chat history is already in the system prompt
            )
            
            # Parse and update profile
            try:
                profile_data = json.loads(json_response)
                await asyncio.to_thread(
                    self.storage.update_user_profile_deep,
                    user_id,
                    profile_data
                )
                logger.info(f"Successfully updated profile for {user_id}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse profile JSON: {e}")
                logger.error(f"Raw response: {json_response}")
            
        except Exception as e:
            logger.error(f"User profiling error: {e}", exc_info=True)