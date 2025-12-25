# ============================================================================
# FILE: utils/tools.py - DSA Optimized Utilities
# ============================================================================
"""Optimized utility functions using data structures."""

import json
import re
from functools import lru_cache
from collections import deque
import heapq
from typing import List, Dict, Optional

# LRU Cache for frequently accessed operations
@lru_cache(maxsize=128)
def json_parser_cached(text: str) -> Dict:
    """Cached JSON parser with fallback strategies."""
    # Remove markdown code blocks
    text = re.sub(r'```json\s*|\s*```', '', text)
    
    # Find JSON blocks
    stack = []
    start = None
    
    for i, ch in enumerate(text):
        if ch in "{[":
            if not stack:
                start = i
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            if (stack[-1] == "{" and ch == "}") or (stack[-1] == "[" and ch == "]"):
                stack.pop()
                if not stack and start is not None:
                    block = text[start:i+1]
                    
                    # Strategy 1: Direct parse
                    try:
                        return json.loads(block)
                    except:
                        pass
                    
                    # Strategy 2: Fix quotes
                    try:
                        fixed = block.replace("'", '"')
                        return json.loads(fixed)
                    except:
                        pass
                    
                    # Strategy 3: Regex cleanup
                    try:
                        cleaned = re.sub(r'(\w+):', r'"\1":', block)
                        return json.loads(cleaned)
                    except:
                        pass
    
    raise ValueError("No valid JSON found")


class MessageBuffer:
    """Deque-based efficient message buffer."""
    
    def __init__(self, maxsize: int = 100):
        self.buffer = deque(maxlen=maxsize)
    
    def add(self, message: Dict):
        """O(1) append operation."""
        self.buffer.append(message)
    
    def get_recent(self, n: int) -> List[Dict]:
        """O(n) retrieval of last n messages."""
        return list(self.buffer)[-n:] if n < len(self.buffer) else list(self.buffer)
    
    def clear(self):
        """O(1) clear operation."""
        self.buffer.clear()


class PriorityTaskQueue:
    """Min-heap based priority queue for task scheduling."""
    
    def __init__(self):
        self.heap = []
        self.counter = 0
    
    def add_task(self, priority: int, task: callable, *args, **kwargs):
        """O(log n) insertion."""
        heapq.heappush(self.heap, (priority, self.counter, task, args, kwargs))
        self.counter += 1
    
    def get_next_task(self) -> Optional[tuple]:
        """O(log n) extraction."""
        if self.heap:
            _, _, task, args, kwargs = heapq.heappop(self.heap)
            return task, args, kwargs
        return None


def chunk_text_optimized(text: str, max_chars: int = 500) -> List[str]:
    """Optimized text chunking using sliding window."""
    if not text or len(text) <= max_chars:
        return [text] if text else []
    
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_len = len(word) + 1
        if current_length + word_len > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_len
        else:
            current_chunk.append(word)
            current_length += word_len
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks[:10]


@lru_cache(maxsize=256)
def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """Cached keyword extraction."""
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'was', 'are', 'were'}
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [w for w in words if len(w) > 3 and w not in stopwords]
    
    # Frequency-based ranking
    freq = {}
    for word in keywords:
        freq[word] = freq.get(word, 0) + 1
    
    sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_keywords[:max_keywords]]