# ============================================================================
# FILE: src/storage.py - Optimized Storage with User Management & RAG (Qdrant + Hybrid Search)
# ============================================================================
"""Storage manager with User Management and Hybrid Search using Qdrant."""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from functools import lru_cache
from pymongo import MongoClient, DESCENDING
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, MatchText, 
    SearchRequest, Prefetch, QueryRequest, FusionQuery
)
from langchain_openai import OpenAIEmbeddings
from utils.config import Settings
import asyncio
import uuid
import os
import gridfs
 
logger = logging.getLogger(__name__)
MONGODB_URL = os.getenv("MONGODB_URL", Settings.MONGO_URL)
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", Settings.MONGO_DB_NAME)

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", Settings.QDRANT_URL)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", Settings.QDRANT_API_KEY)
 
class StorageManager:
    """Optimized storage with User Management and Hybrid Vector Search using Qdrant."""
    
    def __init__(self):
        # MongoDB
        self.client = MongoClient(MONGODB_URL)
        self.db = self.client[MONGO_DB_NAME]
        self.users = self.db.users
        self.sessions = self.db.chat_sessions
        self.messages = self.db.message_history
        self.fs = gridfs.GridFS(self.db)
        
        # Create indexes
        self.users.create_index("user_id", unique=True)
        self.sessions.create_index([("user_id", 1), ("last_updated", DESCENDING)])
        self.sessions.create_index("session_id", unique=True)
        self.messages.create_index([("session_id", 1), ("timestamp", 1)])
        
        # Embeddings
        self.embeddings = OpenAIEmbeddings(model=Settings.EMBEDDING_MODEL)

        # QDRANT VECTOR DB INITIALIZATION ========================================
        self.collection_name = "chat_memory"
        
        try:
            logger.info("|                     STORAGE SERVICES                      |\n|-----------------------------------------------------------|")
            if not QDRANT_URL or not QDRANT_API_KEY:
                logger.warning(".......Qdrant credentials not found in environment variables........🍁")
                self.vector_client = None
            else:
                # Initialize Qdrant client
                self.vector_client = QdrantClient(
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                    timeout=30
                )
                
                # Ensure collection exists with text indexing
                self._ensure_vector_collection()
                logger.info("| QDRANT VECTOR DATABASE              |       DONE          |\n|-----------------------------------------------------------|")
                
        except Exception as e:
            logger.error(f"...........Qdrant initialization failed: {e}..........⚠️")
            self.vector_client = None
        
        logger.info("| STORAGE MANAGER                     |       DONE          |\n|-----------------------------------------------------------|")
    
    def _ensure_vector_collection(self):
        """Create Qdrant collection with full-text indexing for hybrid search."""
        if not self.vector_client:
            logger.warning("Qdrant client not available, skipping collection setup")
            return
            
        try:
            # Get embedding dimension
            test_embedding = self.embeddings.embed_query("test")
            dimension = len(test_embedding)
            
            # Check if collection exists
            collections = self.vector_client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                # Create collection with cosine distance
                self.vector_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f" ......Created Qdrant collection.....'{self.collection_name}'......with dimension {dimension}........🍀")
                
                # Create full-text index on 'content' field for hybrid search
                self.vector_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="content",
                    field_schema="text"  # Enable full-text search
                )
                logger.info("........Created full-text index on 'content' field for hybrid search..........🍀")
                
                # Create keyword index on 'user_id' field for filtering
                self.vector_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="user_id",
                    field_schema="keyword"  # Enable keyword matching for user filtering
                )
                logger.info("........Created keyword index on 'user_id' field for filtering..........🍀")
                
            else:
                # Verify dimension matches
                collection_info = self.vector_client.get_collection(self.collection_name)
                current_dim = collection_info.config.params.vectors.size
                
                if current_dim != dimension:
                    logger.warning(f"................. Dimension mismatch (expected {dimension}, got {current_dim})............🍁")
                    logger.info("Recreating collection with correct dimension...")
                    
                    # Delete and recreate
                    self.vector_client.delete_collection(self.collection_name)
                    self.vector_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=dimension,
                            distance=Distance.COSINE
                        )
                    )
                    
                    # Recreate text index
                    self.vector_client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="content",
                        field_schema="text"
                    )
                    # Create keyword index for user_id
                    self.vector_client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="user_id",
                        field_schema="keyword"
                    )
                    logger.info(f"..........Recreated collection with dimension {dimension} and indexes............🍀")
                else:
                    # Ensure text index exists
                    try:
                        self.vector_client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name="content",
                            field_schema="text"
                        )
                    except Exception:
                        pass  # Index might already exist
                    
                    # Ensure user_id keyword index exists
                    try:
                        self.vector_client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name="user_id",
                            field_schema="keyword"
                        )
                    except Exception:
                        pass  # Index might already exist
                    
        except Exception as e:
            logger.error(f"......... Vector collection setup failed: {e}..........🍁")
            self.vector_client = None

    # ========================================
    # USER MANAGEMENT
    # ========================================
    
    def _initialize_questions_schema(self) -> list:
        """Initialize 8 assessment questions tracking."""
        return [
            {"dimension": "overall_mood", "answered": False, "answered_on": None},
            {"dimension": "life_satisfaction", "answered": False, "answered_on": None},
            {"dimension": "energy_level", "answered": False, "answered_on": None},
            {"dimension": "coping_ability", "answered": False, "answered_on": None},
            {"dimension": "low_mood_stress", "answered": False, "answered_on": None},
            {"dimension": "concentration_clarity", "answered": False, "answered_on": None},
            {"dimension": "social_connection", "answered": False, "answered_on": None},
            {"dimension": "safety_risk", "answered": False, "answered_on": None}
        ]
    
    def register_user(self, user_data: Dict) -> Dict:
        """Register or update a user."""
        user_id = user_data["user_id"]
        user_data["updated_at"] = datetime.utcnow()
        
        # Check if user exists
        existing_user = self.users.find_one({"user_id": user_id})
        
        # Update user data
        self.users.update_one(
            {"user_id": user_id},
            {"$set": user_data},
            upsert=True
        )
        
        # Initialize assessment on first registration (Day 1)
        if not existing_user:
            assessment_data = {
                "first_assessment_started": datetime.utcnow(),
                "current_assessment_active": True,
                "assessment_start_date": datetime.utcnow(),
                "last_assessment_completion_date": None,
                "user_message_count_in_assessment": 0,
                "questions_answered": self._initialize_questions_schema()
            }
            
            self.users.update_one(
                {"user_id": user_id},
                {"$set": {"assessment_data": assessment_data}}
            )
            logger.info(f"Initialized assessment for new user: {user_id}")
        
        return user_data

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user demographics."""
        return self.users.find_one({"user_id": user_id}, {"_id": 0})
    
    def update_user_profile_deep(self, user_id: str, profile_data: Dict):
        """Deep merge extracted profile data into user document."""
        try:
            profile_data.pop("user_id", None)
            
            update_ops = {}
            for key, value in profile_data.items():
                if isinstance(value, dict):
                    for nested_key, nested_value in value.items():
                        if nested_value is not None:
                            update_ops[f"{key}.{nested_key}"] = nested_value
                elif value is not None:
                    update_ops[key] = value
            
            if update_ops:
                self.users.update_one(
                    {"user_id": user_id},
                    {"$set": update_ops},
                    upsert=False
                )
                logger.info(f"Updated profile for user {user_id}")
        except Exception as e:
            logger.error(f".........Profile update error: {e}..........🍁")
    
    # ========================================
    # MENTAL HEALTH ASSESSMENT TRACKING
    # ========================================
    
    def should_trigger_assessment(self, user_id: str) -> bool:
        """Check if assessment should be triggered."""
        try:
            user = self.get_user(user_id)
            if not user:
                return False
            
            assessment_data = user.get("assessment_data", {})
            
            if assessment_data.get("current_assessment_active", False):
                return False
            
            last_completion = assessment_data.get("last_assessment_completion_date")
            
            if not last_completion:
                return True
            
            days_since_completion = (datetime.utcnow() - last_completion).days
            
            if days_since_completion >= 7:
                logger.info(f"..........7-day cooldown complete for {user_id}, triggering new assessment..........🍀")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f".......Error checking assessment trigger: {e}........🍁")
            return False
    
    def create_assessment_marker(self, user_id: str):
        """Mark that assessment period has started."""
        try:
            update_data = {
                "assessment_data.last_assessment_date": datetime.utcnow(),
                "assessment_data.current_assessment_active": True,
                "assessment_data.assessment_start_date": datetime.utcnow()
            }
            
            self.users.update_one(
                {"user_id": user_id},
                {
                    "$set": update_data,
                    "$inc": {"assessment_data.assessment_count": 1}
                }
            )
            logger.info(f".........Created assessment marker for user {user_id}..........🍀")
        except Exception as e:
            logger.error(f"........Error creating assessment marker: {e}..........🍁")
    
    def get_active_assessment_status(self, user_id: str) -> bool:
        """Check if user has active assessment."""
        try:
            user = self.get_user(user_id)
            if not user:
                return False
            
            assessment_data = user.get("assessment_data", {})
            if not assessment_data.get("current_assessment_active", False):
                return False
            
            if self.check_all_questions_answered(user_id):
                logger.info(f"..........All assessment questions answered for {user_id}, completing early.........🍀")
                self.complete_assessment(user_id)
                return False
            
            assessment_start = assessment_data.get("assessment_start_date")
            if assessment_start:
                days_active = (datetime.utcnow() - assessment_start).days
                if days_active >= 30:
                    logger.warning(f".........Assessment timeout (30 days) for {user_id}, auto-completing.........🍀")
                    self.complete_assessment(user_id)
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f".......Error getting assessment status: {e}........🍁")
            return False
    
    def complete_assessment(self, user_id: str):
        """Mark assessment complete, reset for next cycle."""
        try:
            completion_date = datetime.utcnow()
            
            self.users.update_one(
                {"user_id": user_id},
                {"$set": {
                    "assessment_data.current_assessment_active": False,
                    "assessment_data.last_assessment_completion_date": completion_date,
                    "assessment_data.last_assessment_date": completion_date,
                    "assessment_data.user_message_count_in_assessment": 0
                }}
            )
            
            logger.info(f"........Assessment completed for user {user_id} at {completion_date}..........🍀")
        except Exception as e:
            logger.error(f".............Error completing assessment: {e}..........🍁")
    
    def increment_assessment_message_count(self, user_id: str) -> int:
        """Increment and return user message count in current assessment."""
        try:
            result = self.users.find_one_and_update(
                {"user_id": user_id},
                {"$inc": {"assessment_data.user_message_count_in_assessment": 1}},
                return_document=True
            )
            if result:
                return result.get("assessment_data", {}).get("user_message_count_in_assessment", 0)
            return 0
        except Exception as e:
            logger.error(f"......Error incrementing message count: {e}......🍁")
            return 0
    
    def is_assessment_active(self, user_id: str) -> bool:
        """Quick check if assessment is currently active."""
        try:
            user = self.get_user(user_id)
            if not user:
                return False
            return user.get("assessment_data", {}).get("current_assessment_active", False)
        except Exception as e:
            logger.error(f".......Error checking assessment active status: {e}........🍁")
            return False
    
    def get_questions_status(self, user_id: str) -> list:
        """Get current status of all 8 questions."""
        try:
            user = self.get_user(user_id)
            if not user:
                return []
            return user.get("assessment_data", {}).get("questions_answered", [])
        except Exception as e:
            logger.error(f".........Error getting questions status: {e}..........🍁")
            return []
    
    def mark_question_answered(self, user_id: str, dimension: str):
        """Mark a specific dimension as answered."""
        try:
            self.users.update_one(
                {
                    "user_id": user_id,
                    "assessment_data.questions_answered.dimension": dimension
                },
                {"$set": {
                    "assessment_data.questions_answered.$.answered": True,
                    "assessment_data.questions_answered.$.answered_on": datetime.utcnow()
                }}
            )
            logger.info(f"..........Marked dimension '{dimension}' as answered for user {user_id}..........🍀")
        except Exception as e:
            logger.error(f"........Error marking question answered: {e}..........🍁")
    
    def check_all_questions_answered(self, user_id: str) -> bool:
        """Check if all 8 questions have been answered."""
        try:
            user = self.get_user(user_id)
            if not user:
                return False
            
            questions = user.get("assessment_data", {}).get("questions_answered", [])
            return len(questions) == 8 and all(q.get("answered", False) for q in questions)
        except Exception as e:
            logger.error(f"....Error checking all questions answered: {e}.......🍁")
            return False
    
    def start_new_assessment_cycle(self, user_id: str):
        """Start a new assessment cycle after 7-day cooldown."""
        try:
            self.users.update_one(
                {"user_id": user_id},
                {"$set": {
                    "assessment_data.current_assessment_active": True,
                    "assessment_data.assessment_start_date": datetime.utcnow(),
                    "assessment_data.user_message_count_in_assessment": 0,
                    "assessment_data.questions_answered": self._initialize_questions_schema()
                }}
            )
            
            logger.info(f"...........Started new assessment cycle for user {user_id}..........🍀")
        except Exception as e:
            logger.error(f".........Error starting new assessment cycle: {e}..........🍁")
    
    def store_mental_health_report(self, user_id: str, report_data: Dict):
        """Save generated mental health reports."""
        try:
            reports_collection = self.db.mental_health_reports
            
            report_doc = {
                "report_id": f"report_{uuid.uuid4().hex}",
                "user_id": user_id,
                "generated_at": datetime.utcnow(),
                "report_data": report_data
            }
            
            reports_collection.insert_one(report_doc)
            logger.info(f".......Stored mental health report for user {user_id}.......🍀")
        except Exception as e:
            logger.error(f"......Error storing mental health report: {e}......🍁")
    
    def get_latest_report(self, user_id: str) -> Optional[Dict]:
        """Get the most recent mental health report for a user."""
        try:
            reports_collection = self.db.mental_health_reports
            
            latest_report = reports_collection.find_one(
                {"user_id": user_id},
                {"_id": 0},
                sort=[("generated_at", DESCENDING)]
            )
            
            if latest_report:
                return latest_report.get("report_data")
            return None
            
        except Exception as e:
            logger.error(f"........Error retrieving latest report: {e}..........🍁")
            return None
    
    def update_last_report_date(self, user_id: str):
        """Update the last report generation timestamp."""
        try:
            self.users.update_one(
                {"user_id": user_id},
                {"$set": {"last_report_generated_date": datetime.utcnow()}}
            )
            logger.info(f".........Updated last report date for user {user_id}..........🍀")
        except Exception as e:
            logger.error(f".......Error updating last report date: {e}..........🍁")

    # ========================================
    # SESSION OPERATIONS
    # ========================================
    
    def create_session(self, user_id: str, session_id: str) -> str:
        """Create new session."""
        doc = {
            "user_id": user_id,
            "session_id": session_id,
            "session_title": "New chat",
            "created_at": datetime.utcnow(),
            "last_updated": datetime.utcnow()
        }
        self.sessions.insert_one(doc)
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session details."""
        return self.sessions.find_one({"session_id": session_id})

    def get_user_sessions(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get user sessions."""
        cursor = self.sessions.find(
            {"user_id": user_id},
            {"_id": 0}
        ).sort("last_updated", DESCENDING).limit(limit)
        return list(cursor)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        try:
            session = self.sessions.find_one({"session_id": session_id})
            if not session:
                logger.warning(f".......Session {session_id} not found for deletion.......🍁")
                return False
            
            msg_result = self.messages.delete_many({"session_id": session_id})
            logger.info(f".........Deleted {msg_result.deleted_count} messages for session {session_id}..........🍀")
            
            result = self.sessions.delete_one({"session_id": session_id})
            
            logger.info(f"......Deleted session {session_id}........🍀")
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"..........Error deleting session: {e}..........🍁")
            return False

    def update_session_title(self, session_id: str, title: str) -> bool:
        """Update session title/name."""
        try:
            session = self.sessions.find_one({"session_id": session_id})
            if not session:
                logger.warning(f"........Session {session_id} not found.........🍁")
                return False
            
            result = self.sessions.update_one(
                {"session_id": session_id},
                {"$set": {"session_title": title, "last_updated": datetime.utcnow()}}
            )
            logger.info(f"..........Updated session_title for session {session_id}..........🍀")
            return result.matched_count > 0
        except Exception as e:
            logger.error(f".........Error updating session title: {e}..........🍁")
            return False

    # ========================================
    # MESSAGE OPERATIONS
    # ========================================
    
    def save_message(self, session_id: str, user_id: str, role: str, content: str):
        """Save a single message."""
        doc = {
            "session_id": session_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow()
        }
        self.messages.insert_one(doc)
        
        self.sessions.update_one(
            {"session_id": session_id},
            {"$set": {"last_updated": datetime.utcnow()}}
        )

    def get_recent_messages(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get recent messages for context."""
        cursor = self.messages.find(
            {"session_id": session_id},
            {"_id": 0, "role": 1, "content": 1, "timestamp": 1}
        ).sort("timestamp", DESCENDING).limit(limit)
        
        return list(cursor)[::-1]
    
    def count_user_messages(self, user_id: str) -> int:
        """Count total messages for a user."""
        return self.messages.count_documents({"user_id": user_id})
    
    def get_chat_history_for_analysis(self, user_id: str, limit: int = 30) -> str:
        """Get formatted chat history for LLM analysis."""
        cursor = self.messages.find(
            {"user_id": user_id},
            {"_id": 0, "role": 1, "content": 1}
        ).sort("timestamp", DESCENDING).limit(limit)
        
        messages = list(cursor)[::-1]
        
        formatted = []
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        
        return "\n\n".join(formatted)
    
    def get_user_recent_conversations(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get the most recent conversations for a user."""
        cursor = self.messages.find(
            {"user_id": user_id},
            {"_id": 0, "role": 1, "content": 1, "timestamp": 1, "session_id": 1}
        ).sort("timestamp", DESCENDING).limit(limit * 2)
        
        messages = list(cursor)[::-1]
        return messages

    # ========================================
    # VECTOR MEMORY (RAG) - QDRANT WITH HYBRID SEARCH
    # ========================================
    
    async def vectorize_and_store(self, user_id: str, user_text: str, bot_text: str):
        """
        Vectorize the interaction and store in Qdrant with metadata for hybrid search.
        
        CRITICAL FIX: We now embed ONLY the user_text (not the conversational pair)
        to match the format of retrieval queries. This dramatically improves semantic
        similarity scores since query and storage formats are now consistent.
        
        Storage format: "I work as a software engineer at Google"
        Query format: "Where do I work?"
        → Much higher semantic similarity!
        """
        if not self.vector_client:
            logger.warning(".......Qdrant unavailable, skipping vector storage.......🍁")
            return
            
        try:
            # FIXED: Embed ONLY user text to match query format
            # This is the key fix - queries are single statements, not conversations
            embedding_text = user_text  # Just the user message!
            
            # Full conversation for metadata/keyword search
            full_content = f"User: {user_text}\nAssistant: {bot_text}"
            
            # Generate embedding from user text ONLY
            embedding = await asyncio.to_thread(
                self.embeddings.embed_query,
                embedding_text  # Changed from 'content' to 'embedding_text'
            )
            
            # Extract keywords from full conversation for hybrid search
            keywords = self._extract_keywords(full_content)
            
            # Create point with rich metadata
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,  # Embedding from user_text only
                payload={
                    "user_id": user_id,
                    "content": full_content,  # Store full conversation for display
                    "user_text": user_text,
                    "bot_text": bot_text,
                    "keywords": keywords,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # Insert into Qdrant
            self.vector_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            logger.info(f"......Stored vector memory for user {user_id} (user-text embedding)..........🍀")
            
        except Exception as e:
            logger.error(f"...........Vector storage error (non-fatal): {e}..........🍁")

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords from text for enhanced hybrid search.
        Simple implementation - can be enhanced with NLP libraries.
        """
        # Mental health related keywords to boost
        important_terms = [
            "anxiety", "depression", "stress", "panic", "worry", "fear",
            "sad", "happy", "angry", "frustrated", "overwhelmed", "tired",
            "sleep", "insomnia", "energy", "motivation", "therapy", "medication",
            "work", "family", "relationship", "friend", "job", "school"
        ]
        
        text_lower = text.lower()
        found_keywords = [term for term in important_terms if term in text_lower]
        
        return found_keywords

    async def search_memory(
        self, 
        user_id: str, 
        query: str, 
        limit: int = 3,
        mode: str = "hybrid"
    ) -> str:
        """
        Hybrid search for relevant past interactions.
        
        Args:
            user_id: User ID to filter results
            query: Search query
            limit: Number of results to return
            mode: Search mode - "semantic", "keyword", or "hybrid"
        
        Returns:
            Formatted string of relevant past conversations
        """
        if not self.vector_client:
            logger.debug("........Qdrant unavailable, returning empty memory context.........🍁")
            return ""
            
        try:
            if mode == "keyword":
                # Keyword-only search using text matching
                results = self._keyword_search(user_id, query, limit)
            elif mode == "semantic":
                # Semantic-only search using vector similarity
                results = await self._semantic_search(user_id, query, limit)
            else:  # hybrid (default)
                # Combined semantic + keyword search
                results = await self._hybrid_search(user_id, query, limit)
            
            if not results:
                logger.info(f"..........Memory search returned 0 results for user {user_id}..........🍁")
                return ""
            
            # Format memories
            memories = []
            filtered_count = 0
            for result in results:
                payload = result.payload
                score = result.score
                
                # Filter low relevance results
                if score < 0.4:
                    filtered_count += 1
                    logger.debug(f"   Filtered result (score {score:.2f} < 0.4): {payload.get('content', '')[:50]}...")
                    continue
                
                content = payload.get("content", "")
                timestamp = payload.get("timestamp", "")
                
                memories.append(f"[Relevance: {score:.2f}] {content}")
            
            logger.info(f"..........Memory search: Found {len(results)} results, {filtered_count} filtered out, {len(memories)} passed threshold..........🍀")
            
            if not memories:
                logger.warning(f"..........No memories passed relevance threshold (0.4) for query: '{query[:50]}'..........🍁")
                return ""
            
            logger.info(f"............Retrieved {len(memories)} relevant memories using {mode} search.........🍀")
            return "\n---\n".join(memories)
            
        except Exception as e:
            logger.error(f"........Memory search error (non-fatal): {e}.......🍁")
            return ""

    async def _semantic_search(self, user_id: str, query: str, limit: int) -> List:
        """Pure semantic search using vector similarity."""
        try:
            # Generate query embedding
            query_embedding = await asyncio.to_thread(
                self.embeddings.embed_query,
                query
            )
            
            # Search with user filter
            results = self.vector_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        )
                    ]
                ),
                limit=limit
            )
            
            if hasattr(results, 'points'):
                return results.points
            return results
                        
        except Exception as e:
            logger.error(f"..........Semantic search error: {e}..........🍁")
            return []

    def _keyword_search(self, user_id: str, query: str, limit: int) -> List:
        """Pure keyword search using text matching."""
        try:
            # Extract keywords from query
            query_keywords = query.lower().split()
            
            # Search with text matching on content field
            results = self.vector_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        ),
                        FieldCondition(
                            key="content",
                            match=MatchText(text=query)
                        )
                    ]
                ),
                limit=limit
            )
            
            # Convert scroll results to search-like format
            points = results[0] if results else []
            
            # Create mock search results with scores
            formatted_results = []
            for point in points[:limit]:
                # Simple scoring based on keyword matches
                content = point.payload.get("content", "").lower()
                score = sum(1 for keyword in query_keywords if keyword in content) / max(len(query_keywords), 1)
                
                # Mock result object
                class MockResult:
                    def __init__(self, payload, score):
                        self.payload = payload
                        self.score = score
                
                formatted_results.append(MockResult(point.payload, score))
            
            return formatted_results
            
        except Exception as e:
            logger.error(f".........Keyword search error: {e}.........🍁")
            return []

    async def _hybrid_search(self, user_id: str, query: str, limit: int) -> List:
        """
        Hybrid search combining semantic and keyword approaches.
        Matches query keywords against stored database content.
        """
        try:
            # Generate query embedding for semantic search
            query_embedding = await asyncio.to_thread(
                self.embeddings.embed_query,
                query
            )
            
            # Extract keywords from USER QUERY
            query_keywords = self._extract_keywords(query)
            query_lower = query.lower()
            
            # Build base filter (user_id only)
            filter_conditions = [
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id)
                )
            ]
            
            # If query has keywords, add full-text search filter
            # This searches the stored 'content' field in the database
            if query_keywords or len(query.split()) > 0:
                filter_conditions.append(
                    FieldCondition(
                        key="content",
                        match=MatchText(text=query)  # Searches DB content for query terms
                    )
                )
            
            # Perform hybrid search (semantic + keyword filter)
            results = self.vector_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=Filter(must=filter_conditions),
                limit=limit * 3  # Get more results for better post-filtering
            )
            
            # Post-process: boost results based on keyword matches in DB content
            results = results.points if hasattr(results, 'points') else results
            scored_results = []
            for result in results:
                base_score = result.score
                db_content = result.payload.get("content", "").lower()
                db_keywords = result.payload.get("keywords", [])
                
                # Count how many query keywords appear in DB content
                query_words = query_lower.split()
                
                # Match 1: Query keywords found in stored keywords list
                keyword_match_count = sum(1 for kw in query_keywords if kw in db_keywords)
                
                # Match 2: Query words found in stored content text
                content_match_count = sum(1 for word in query_words if word in db_content and len(word) > 2)
                
                # Calculate boost: +0.1 per keyword match, +0.05 per content word match
                keyword_boost = (keyword_match_count * 0.15) + (content_match_count * 0.05)
                hybrid_score = base_score + keyword_boost
                
                # Create boosted result
                class BoostedResult:
                    def __init__(self, payload, score, match_info):
                        self.payload = payload
                        self.score = min(score, 1.0)  # Cap at 1.0
                        self.match_info = match_info
                
                match_info = {
                    "keyword_matches": keyword_match_count,
                    "content_matches": content_match_count,
                    "boost_applied": keyword_boost
                }
                
                scored_results.append(BoostedResult(result.payload, hybrid_score, match_info))
            
            # Sort by hybrid score and return top results
            scored_results.sort(key=lambda x: x.score, reverse=True)
            
            # Log matching info for top result
            if scored_results:
                top = scored_results[0]
                logger.debug(
                    f"....Top hybrid match - Score: {top.score:.3f}, "
                    f"Keyword matches: {top.match_info['keyword_matches']}, "
                    f"Content matches: {top.match_info['content_matches']}..........🍁"
                )
            
            return scored_results[:limit]
            
        except Exception as e:
            logger.error(f"........Hybrid search error: {e}.........🍁")
            # Fallback to semantic search
            return await self._semantic_search(user_id, query, limit)

    # ========================================
    # USER DELETION (SOFT DELETE)
    # ========================================
    
    def soft_delete_user(self, user_id: str) -> Dict:
        """
        Soft delete a user by moving all their data to deleted_users collection.
        Also deletes vector embeddings from Qdrant.
        """
        try:
            # Step 1: Check if user exists
            user = self.get_user(user_id)
            if not user:
                return {
                    "status": "error",
                    "message": f"User {user_id} not found",
                    "user_id": user_id
                }
            
            # Step 2: Check if already deleted
            deleted_users_collection = self.db.deleted_users
            existing_deletion = deleted_users_collection.find_one({"user_id": user_id})
            if existing_deletion:
                return {
                    "status": "error",
                    "message": f"User {user_id} is already deleted",
                    "user_id": user_id,
                    "deleted_at": existing_deletion.get("deleted_at")
                }
            
            # Step 3: Gather all user data
            logger.info(f"........Starting soft delete for user {user_id}.........🍀")
            
            sessions = list(self.sessions.find({"user_id": user_id}, {"_id": 0}))
            messages = list(self.messages.find({"user_id": user_id}, {"_id": 0}))
            
            reports_collection = self.db.mental_health_reports
            reports = list(reports_collection.find({"user_id": user_id}, {"_id": 0}))
            
            audio_files = list(self.fs.find({"user_id": user_id}))
            audio_file_ids = [
                {"file_id": str(f._id), "filename": f.filename, "upload_date": f.upload_date} 
                for f in audio_files
            ]
            
            # Count vector embeddings from Qdrant
            vector_count = 0
            if self.vector_client:
                try:
                    results = self.vector_client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=Filter(
                            must=[
                                FieldCondition(
                                    key="user_id",
                                    match=MatchValue(value=user_id)
                                )
                            ]
                        ),
                        limit=10000
                    )
                    vector_count = len(results[0]) if results and results[0] else 0
                except Exception as e:
                    logger.warning(f".......Could not count vector embeddings: {e}.......🍁")
            
            # Step 4: Create archived document
            deleted_user_doc = {
                "user_id": user_id,
                "deleted_at": datetime.utcnow(),
                "user_data": user,
                "sessions": sessions,
                "messages": messages,
                "reports": reports,
                "audio_files": audio_file_ids,
                "vector_embedding_count": vector_count,
                "deletion_metadata": {
                    "reason": "user_requested",
                    "processed_by": "api",
                    "total_sessions": len(sessions),
                    "total_messages": len(messages),
                    "total_reports": len(reports),
                    "total_audio_files": len(audio_file_ids)
                }
            }
            
            # Step 5: Insert into deleted_users collection
            deleted_users_collection.insert_one(deleted_user_doc)
            logger.info(f"........Archived user data for {user_id} in deleted_users collection.........🍀")
            
            # Step 6: Delete vector embeddings from Qdrant
            if self.vector_client:
                try:
                    self.vector_client.delete(
                        collection_name=self.collection_name,
                        points_selector=Filter(
                            must=[
                                FieldCondition(
                                    key="user_id",
                                    match=MatchValue(value=user_id)
                                )
                            ]
                        )
                    )
                    logger.info(f"........Deleted {vector_count} vector embeddings from Qdrant for user {user_id}.........🍀")
                except Exception as e:
                    logger.error(f".............Error deleting vector embeddings: {e}.........🍁")
            
            # Step 7: Remove from active collections
            reports_deleted = reports_collection.delete_many({"user_id": user_id})
            messages_deleted = self.messages.delete_many({"user_id": user_id})
            sessions_deleted = self.sessions.delete_many({"user_id": user_id})
            user_deleted = self.users.delete_one({"user_id": user_id})
            
            logger.info(f"...........Soft delete completed for user {user_id}..........🍀")
            
            # Step 8: Return summary
            return {
                "status": "success",
                "message": f"User {user_id} has been soft deleted successfully",
                "user_id": user_id,
                "deleted_at": deleted_user_doc["deleted_at"].isoformat(),
                "summary": {
                    "sessions_archived": len(sessions),
                    "messages_archived": len(messages),
                    "reports_archived": len(reports),
                    "audio_files_referenced": len(audio_file_ids),
                    "vector_embeddings_deleted": vector_count
                },
                "note": "Data archived in deleted_users collection. Vector embeddings deleted from Qdrant."
            }
            
        except Exception as e:
            logger.error(f"...........Error during soft delete for user {user_id}: {e}........🍁", exc_info=True)
            return {
                "status": "error",
                "message": f"An error occurred during soft delete: {str(e)}",
                "user_id": user_id
            }

    # ========================================
    # CLEANUP
    # ========================================

    def close(self):
        """Close database connections."""
        self.client.close()
        if self.vector_client:
            self.vector_client.close()

    def store_audio(self, user_id: str, session_id: str, audio_bytes: bytes, format: str):
        """Store audio file in GridFS."""
        try:
            filename = f"{user_id}_{session_id}_{datetime.utcnow().isoformat()}.{format}"
            self.fs.put(
                audio_bytes,
                filename=filename,
                user_id=user_id,
                session_id=session_id,
                content_type=f"audio/{format}",
                timestamp=datetime.utcnow()
            )
            logger.info(f".........Stored audio: {filename}..........🍀")
        except Exception as e:
            logger.error(f".......Audio storage error: {e}.......🍁")