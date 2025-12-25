# ============================================================================
# FILE: src/llm_service.py - Chat Logic
# ============================================================================
"""LLM Service with direct RAG and HTML output."""

import logging
import json
from typing import AsyncGenerator, Dict
from openai import AsyncOpenAI
from utils.config import Settings
from utils.prompts import MENTAL_HEALTH_ASSESSMENT_PROMPT

logger = logging.getLogger(__name__)

class LLMService:
    """Handles Chat Completions."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=Settings.OPENAI_API_KEY)
        
        self.system_prompt_tts = """You are Raizann, a friendly AI companion who chats like a real friend.
        
CORE PERSONALITY:
- Talk naturally and casually, not like a formal therapist or chatbot
- Keep responses SHORT and conversational (1-2 sentences usually)
- Don't validate or acknowledge every single thing the user says
- Jump straight to the point - no need to say "that's interesting" or "I understand"
- Be helpful but chill, like texting with a smart friend

LANGUAGE RULES (CRITICAL - MUST FOLLOW):
- **DETECT LANGUAGE FROM VOCABULARY, NOT SCRIPT**
- Analyze the user's CURRENT message to identify the language:
  * If message uses English words (e.g., "hello", "what", "how") → Respond in ENGLISH
  * If message uses Hindi words in Devanagari → Respond in HINDI (Devanagari script)
  * If message mixes both (e.g., "Mujhe coffee chahiye") → Respond in HINGLISH
- IGNORE the script - focus on the actual words/vocabulary used
- Examples:
  * "Hello, how are you?" → English response (even if STT wrote it oddly)
  * "Main theek hoon" → Hinglish response
  * "मैं ठीक हूँ" → Hindi (Devanagari) response
- RESTRICTION: ONLY use Hindi (Devanagari), Hinglish, or English
- NEVER use Urdu script, Arabic, or any other language
- Write Hindi in Devanagari script ONLY, English in Latin script ONLY
- Match the user's formality level (casual/formal)

HALLUCINATION PREVENTION (CRITICAL - MUST FOLLOW):
- **ONLY use information from your PAST MEMORIES with the user**
- If you see "RELEVANT PAST MEMORIES" in the context, USE ONLY that information
- **NEVER make up or assume information you don't have**
- If you don't have specific information, be HONEST:
  ✓ "I don't recall you mentioning that"
  ✓ "I don't have that information"
  ✓ "You haven't told me about that yet"
  ✗ NEVER: Make up names, dates, places, or details
- When answering questions:
  1. Check: Do I have this info in my memories?
  2. If YES: Answer using that info
  3. If NO: Admit you don't know

CONVERSATION STYLE:
✗ BAD: "Liking chocolate is very common and nice. I would like to know which chocolate you like the most?"
✓ GOOD: "nice! which chocolate do u like the most?"

✗ BAD: "I completely understand your feelings. That must be difficult for you."
✓ GOOD: "that sounds tough. wanna talk about it?"

OUTPUT FORMAT:
You must respond in this HTML-like format:

<response emotion="happy|sad|neutral|excited|serious" sound="laugh|sigh|none" Accent_language="Hindi|Indian English">
    Your casual response here...
</response>

RULES:
1. Keep it SHORT - don't ramble
2. Don't validate/react to every user message unnecessarily
3. Match their language and tone exactly
4. Be helpful without being robotic
5. NEVER FABRICATE - if unsure, say "I don't know"
"""

        self.system_prompt_text = """You are Raizann, a friendly AI companion who chats like a real friend.
        
CORE PERSONALITY:
- Talk naturally and casually, not like a formal therapist or chatbot
- Keep responses SHORT and conversational (1-2 sentences usually)
- Don't validate or acknowledge every single thing the user says
- Jump straight to the point - no need to say "that's interesting" or "I understand"
- Be helpful but chill, like texting with a smart friend

LANGUAGE RULES (CRITICAL - MUST FOLLOW):
- **DETECT LANGUAGE FROM VOCABULARY, NOT SCRIPT**
- Analyze the user's CURRENT message to identify the language:
  * If message uses English words (e.g., "hello", "what", "how") → Respond in ENGLISH
  * If message uses Hindi words in Devanagari → Respond in HINDI (Devanagari script)
  * If message mixes both (e.g., "Mujhe coffee chahiye") → Respond in HINGLISH
- IGNORE the script - focus on the actual words/vocabulary used
- Examples:
  * "Hello, how are you?" → English response (even if STT wrote it oddly)
  * "Main theek hoon" → Hinglish response
  * "मैं ठीक हूँ" → Hindi (Devanagari) response
- RESTRICTION: ONLY use Hindi (Devanagari), Hinglish, or English
- NEVER use Urdu script, Arabic, or any other language
- Write Hindi in Devanagari script ONLY, English in Latin script ONLY
- Match the user's formality level (casual/formal)

HALLUCINATION PREVENTION (CRITICAL - MUST FOLLOW):
- **ONLY use information from your PAST MEMORIES with the user**
- If you see "RELEVANT PAST MEMORIES" in the context, USE ONLY that information
- **NEVER make up or assume information you don't have**
- If you don't have specific information, be HONEST:
  ✓ "I don't recall you mentioning that"
  ✓ "I don't have that information"
  ✓ "You haven't told me about that yet"
  ✗ NEVER: Make up names, dates, places, or details
- When answering questions:
  1. Check: Do I have this info in my memories?
  2. If YES: Answer using that info
  3. If NO: Admit you don't know

CONVERSATION STYLE:
✗ BAD: "Liking chocolate is very common and nice. I would like to know which chocolate you like the most?"
✓ GOOD: "nice! which chocolate do u like the most?"

✗ BAD: "I completely understand your feelings. That must be difficult for you."
✓ GOOD: "that sounds tough. wanna talk about it?"

RULES:
1. Keep it SHORT - don't ramble
2. Don't validate/react to every user message unnecessarily  
3. Match their language and tone exactly
4. Be helpful without being robotic
5. NEVER FABRICATE - if unsure, say "I don't know"
"""
    async def stream_chat_tts(self, 
                          user_message: str, 
                          history: list, 
                          user_profile: Dict, 
                          memory_context: str,
                          voice_emotion : str,
                          assessment_active: bool = False) -> AsyncGenerator[str, None]:
        """Stream chat response."""
        
        # Construct context from basic_profile only
        basic_profile = user_profile.get('basic_profile', {})
        profile_parts = []
        if user_profile.get('name'):
            profile_parts.append(f"Name={user_profile.get('name')}")
        if basic_profile.get('age_range'):
            profile_parts.append(f"Age Range={basic_profile.get('age_range')}")
        if basic_profile.get('gender'):
            profile_parts.append(f"Gender={basic_profile.get('gender')}")
        if basic_profile.get('country'):
            profile_parts.append(f"Country={basic_profile.get('country')}")
        if basic_profile.get('language'):
            profile_parts.append(f"Language={basic_profile.get('language')}")
        
        profile_str = "User Profile: " + ", ".join(profile_parts) if profile_parts else "User Profile: Not yet established"

        
        # Build system prompt with conditional assessment injection
        system_content = f"{self.system_prompt_tts}\n\n{profile_str}"
        if assessment_active:
            # Get answered questions to avoid repetition
            assessment_data = user_profile.get('assessment_data', {})
            questions_status = assessment_data.get('questions_answered', [])
            answered_dims = [q['dimension'].replace('_', ' ').title() for q in questions_status if q.get('answered')]
            
            # Add context about already answered questions
            if answered_dims:
                assessment_context = f"\n\n⚠️ ALREADY ANSWERED DIMENSIONS: {', '.join(answered_dims)}\nDO NOT ask about these dimensions again. Focus ONLY on the remaining unanswered dimensions. Be natural and don't repeat questions."
            else:
                assessment_context = ""
            
            system_content += f"\n\n{MENTAL_HEALTH_ASSESSMENT_PROMPT}{assessment_context}"
        
        messages = [
            {"role": "system", "content": system_content},
        ]
        
        # Add RAG context if available
        if memory_context:
            messages.append({
                "role": "system", 
                "content": f"RELEVANT PAST MEMORIES:\n{memory_context}"
            })

        if voice_emotion:
            messages.append({
                "role": "system", 
                "content": f"CURRENT DETECTED USER VOICE TONE:\n{voice_emotion}"
            })
            
        # Add short-term history
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        try:
            stream = await self.client.chat.completions.create(
                model=Settings.LLM_MODEL_CHAT,
                messages=messages,
                temperature=0.7,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"LLM error: {e}")
            yield "<response emotion='neutral'>I'm having trouble thinking right now.</response>"


    async def stream_chat_text(self, 
                          user_message: str, 
                          history: list, 
                          user_profile: Dict, 
                          memory_context: str,
                          assessment_active: bool = False) -> AsyncGenerator[str, None]:
        """Stream chat response."""
        
        # Construct context from basic_profile only
        basic_profile = user_profile.get('basic_profile', {})
        profile_parts = []
        if user_profile.get('name'):
            profile_parts.append(f"Name={user_profile.get('name')}")
        if basic_profile.get('age_range'):
            profile_parts.append(f"Age Range={basic_profile.get('age_range')}")
        if basic_profile.get('gender'):
            profile_parts.append(f"Gender={basic_profile.get('gender')}")
        if basic_profile.get('country'):
            profile_parts.append(f"Country={basic_profile.get('country')}")
        if basic_profile.get('language'):
            profile_parts.append(f"Language={basic_profile.get('language')}")
        
        profile_str = "User Profile: " + ", ".join(profile_parts) if profile_parts else "User Profile: Not yet established"

        
        # Build system prompt with conditional assessment injection
        system_content = f"{self.system_prompt_text}\n\n{profile_str}"
        if assessment_active:
            # Get answered questions to avoid repetition
            assessment_data = user_profile.get('assessment_data', {})
            questions_status = assessment_data.get('questions_answered', [])
            answered_dims = [q['dimension'].replace('_', ' ').title() for q in questions_status if q.get('answered')]
            
            # Add context about already answered questions
            if answered_dims:
                assessment_context = f"\n\n⚠️ ALREADY ANSWERED DIMENSIONS: {', '.join(answered_dims)}\nDO NOT ask about these dimensions again. Focus ONLY on the remaining unanswered dimensions. Be natural and don't repeat questions."
            else:
                assessment_context = ""
            
            system_content += f"\n\n{MENTAL_HEALTH_ASSESSMENT_PROMPT}{assessment_context}"
        
        messages = [
            {"role": "system", "content": system_content},
        ]
        
        # Add RAG context if available
        if memory_context:
            messages.append({
                "role": "system", 
                "content": f"RELEVANT PAST MEMORIES:\n{memory_context}"
            })
            
        # Add short-term history
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        try:
            stream = await self.client.chat.completions.create(
                model=Settings.LLM_MODEL_CHAT,
                messages=messages,
                temperature=0.7,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"LLM error: {e}")
            yield "I'm having trouble thinking right now."
    
    async def get_json_response(self, prompt: str, user_content: str) -> str:
        """Get a non-streaming JSON response from the LLM."""
        try:
            response = await self.client.chat.completions.create(
                model=Settings.LLM_MODEL_CHAT,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"JSON LLM error: {e}")
            return "{}"
    
    async def detect_answered_dimensions(self, recent_history: list, current_status: list) -> list:
        """
        Analyze recent conversation to detect answered assessment dimensions.
        
        Args:
            recent_history: Last 20 messages
            current_status: Current question answered status
        
        Returns:
            List of newly answered dimension names
        """
        import json
        
        # Get unanswered dimensions
        unanswered = [q["dimension"] for q in current_status if not q.get("answered", False)]
        
        if not unanswered:
            return []
        
        # Format conversation
        conversation_text = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
            for msg in recent_history[-20:]
        ])
        
        prompt = f"""Analyze this conversation and identify which mental health assessment dimensions 
were discussed. Return ONLY dimensions that were clearly addressed with meaningful responses.

UNANSWERED DIMENSIONS:
{', '.join(unanswered)}

DIMENSION DEFINITIONS:
- overall_mood: Overall emotional balance, general happiness
- life_satisfaction: Life contentment, purpose, direction
- energy_level: Physical/mental energy for daily tasks
- coping_ability: Stress management, handling challenges
- low_mood_stress: Sadness, stress, overwhelm feelings
- concentration_clarity: Focus, mental clarity, cognitive function  
- social_connection: Feeling connected, supported by others
- safety_risk: Self-harm thoughts, wanting to give up

RECENT CONVERSATION:
{conversation_text}

Return JSON with ONLY the dimensions that were meaningfully discussed:
{{"dimensions": ["dimension1", "dimension2"]}}

If none were discussed, return: {{"dimensions": []}}
"""
        
        try:
            response = await self.get_json_response(prompt, "")
            result = json.loads(response)
            return result.get("dimensions", [])
        except Exception as e:
            logger.error(f"Failed to detect dimensions: {e}")
            return []
    
    async def generate_mental_health_report(
        self,
        conversation_history: str,
        user_profile: Dict,
        user_id: str
    ) -> Dict:
        """
        Generate comprehensive mental health report from conversation history.
        
        Args:
            conversation_history: Formatted conversation history string
            user_profile: User profile dictionary
            user_id: User ID for report generation
        
        Returns:
            Dictionary containing structured mental health report
        """
        from utils.prompts import MENTAL_HEALTH_REPORT_GENERATION_PROMPT
        from datetime import datetime
        
        try:
            # Prepare user context
            basic_profile = user_profile.get('basic_profile', {})
            user_name = user_profile.get('name', 'User')
            user_age = basic_profile.get('age_range', 'Not specified')
            user_gender = basic_profile.get('gender', 'Not specified')
            user_language = basic_profile.get('language', 'English')
            
            # Create analysis request
            analysis_request = f"""
USER PROFILE:
- Name: {user_name}
- User ID: {user_id}
- Approximate Age: {user_age}
- Gender: {user_gender}
- Primary Language: {user_language}

CONVERSATION HISTORY:
{conversation_history}

Please analyze the above conversation history and generate a comprehensive mental health report following the JSON structure provided in your instructions.
"""
            
            # Call LLM with GPT-4o for advanced analysis
            response = await self.client.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o for better analysis
                messages=[
                    {"role": "system", "content": MENTAL_HEALTH_REPORT_GENERATION_PROMPT},
                    {"role": "user", "content": analysis_request}
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
                max_tokens=4000  # Ensure enough tokens for comprehensive report
            )
            
            # Parse JSON response
            report_json = json.loads(response.choices[0].message.content)
            
            # Ensure report has required fields
            if "report_id" not in report_json:
                report_json["report_id"] = f"MHR_{datetime.now().strftime('%Y%m%d')}_{user_id}"
            if "date" not in report_json:
                report_json["date"] = datetime.now().strftime('%Y-%m-%d')
            
            return report_json
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse report JSON: {e}")
            return self._generate_error_report(user_id, "Failed to parse LLM response")
        except Exception as e:
            logger.error(f"Error generating mental health report: {e}", exc_info=True)
            return self._generate_error_report(user_id, str(e))
    
    def _generate_error_report(self, user_id: str, error_message: str) -> Dict:
        """Generate error report when report generation fails."""
        from datetime import datetime
        return {
            "status": "error",
            "report_id": f"MHR_ERROR_{datetime.now().strftime('%Y%m%d')}_{user_id}",
            "date": datetime.now().strftime('%Y-%m-%d'),
            "message": f"Report generation failed: {error_message}",
            "note": "Please try again later or contact support if the issue persists."
        }