# ============================================================================
# FILE: src/guest_storage.py - Guest Session Management
# ============================================================================
"""Lightweight in-memory storage for guest chat sessions."""

import logging
import uuid
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque

logger = logging.getLogger(__name__)


class GuestStorageManager:
    """Manages temporary guest chat sessions in memory."""
    
    def __init__(self, max_messages_per_guest: int = 10):
        """
        Initialize guest storage manager.
        
        Args:
            max_messages_per_guest: Maximum messages to keep per guest (moving window)
        """
        self.sessions: Dict[str, Dict] = {}
        self.max_messages = max_messages_per_guest
        self.lock = threading.Lock()
        logger.info("| GUESTSTORAGE MANAGER                |       DONE          |\n|-----------------------------------------------------------|")
    
    def create_guest_session(self) -> str:
        """
        Create a new guest session.
        
        Returns:
            Unique guest ID
        """
        guest_id = f"guest_{uuid.uuid4().hex}"
        
        with self.lock:
            self.sessions[guest_id] = {
                "messages": deque(maxlen=self.max_messages),
                "created_at": datetime.utcnow(),
                "last_activity": datetime.utcnow(),
                "message_count": 0
            }
        
        logger.info(f"Created guest session: {guest_id}")
        return guest_id
    
    def get_guest_messages(self, guest_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Fetch messages for a guest session.
        
        Args:
            guest_id: Guest session ID
            limit: Optional limit on number of messages (defaults to all)
        
        Returns:
            List of message dictionaries with role and content
        """
        with self.lock:
            if guest_id not in self.sessions:
                logger.warning(f"Guest session not found: {guest_id}")
                return []
            
            messages = list(self.sessions[guest_id]["messages"])
            
            if limit and limit < len(messages):
                messages = messages[-limit:]
            
            return messages
    
    def save_guest_message(self, guest_id: str, role: str, content: str):
        """
        Save a message to guest session (with moving window).
        
        Args:
            guest_id: Guest session ID
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        with self.lock:
            if guest_id not in self.sessions:
                logger.error(f"Cannot save message - guest session not found: {guest_id}")
                return
            
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Deque automatically handles moving window (drops oldest when full)
            self.sessions[guest_id]["messages"].append(message)
            self.sessions[guest_id]["last_activity"] = datetime.utcnow()
            
            if role == "user":
                self.sessions[guest_id]["message_count"] = self.sessions[guest_id].get("message_count", 0) + 1
            
            logger.debug(f"Saved message for guest {guest_id} (total: {len(self.sessions[guest_id]['messages'])})")
    
    def delete_guest_session(self, guest_id: str) -> bool:
        """
        Delete a guest session and all its data.
        
        Args:
            guest_id: Guest session ID
        
        Returns:
            True if deleted, False if not found
        """
        with self.lock:
            if guest_id in self.sessions:
                del self.sessions[guest_id]
                logger.info(f"Deleted guest session: {guest_id}")
                return True
            else:
                logger.warning(f"Cannot delete - guest session not found: {guest_id}")
                return False
    
    def cleanup_inactive_sessions(self, inactivity_threshold_minutes: int = 60):
        """
        Remove guest sessions inactive for specified duration.
        
        Args:
            inactivity_threshold_minutes: Minutes of inactivity before cleanup
        """
        threshold = datetime.utcnow() - timedelta(minutes=inactivity_threshold_minutes)
        
        with self.lock:
            inactive_guests = [
                guest_id 
                for guest_id, session in self.sessions.items()
                if session["last_activity"] < threshold
            ]
            
            for guest_id in inactive_guests:
                del self.sessions[guest_id]
                logger.info(f"Auto-cleaned inactive guest session: {guest_id}")
            
            if inactive_guests:
                logger.info(f"Cleaned up {len(inactive_guests)} inactive guest session(s)")
    
    def get_session_count(self) -> int:
        """Get total number of active guest sessions."""
        with self.lock:
            return len(self.sessions)
    
    def session_exists(self, guest_id: str) -> bool:
        """Check if a guest session exists."""
        with self.lock:
            return guest_id in self.sessions

    def can_send_message(self, guest_id: str, max_queries: int = 15) -> bool:
        """
        Check if guest can send more messages.
        
        Args:
            guest_id: Guest session ID
            max_queries: Maximum allowed user queries
            
        Returns:
            True if allowed, False if limit reached
        """
        with self.lock:
            if guest_id not in self.sessions:
                return False
            
            # Get current count (default to 0 if missing)
            count = self.sessions[guest_id].get("message_count", 0)
            return count < max_queries
