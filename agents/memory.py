"""Conversation Memory: Short-term and long-term."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import json
from agents.vector_db import VectorDB
from utils.logging import logger

@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    reasoning: Optional[str] = None
    tool_calls: Optional[List[str]] = None

class ShortTermMemory:
    """In-context memory for current conversation."""
    
    def __init__(self, max_turns: int = 5, max_tokens: int = 2000):
        """
        Initialize short-term memory.
        
        Args:
            max_turns: Maximum number of turns to keep
            max_tokens: Maximum tokens to use in context
        """
        self.messages: List[Message] = []
        self.max_turns = max_turns
        self.max_tokens = max_tokens
    
    def add_message(self, role: str, content: str, reasoning: str = None, tool_calls: List[str] = None) -> None:
        """Add a message to memory."""
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            reasoning=reasoning,
            tool_calls=tool_calls
        )
        self.messages.append(message)
        logger.debug(f"Added {role} message to short-term memory")
        
        # Trim if exceeding max turns
        if len(self.messages) > self.max_turns * 2:  # *2 because user + assistant
            self.messages = self.messages[-(self.max_turns * 2):]
    
    def get_context(self) -> str:
        """Get formatted context for LLM."""
        if not self.messages:
            return ""
        
        context_parts = []
        for msg in self.messages[-self.max_turns * 2:]:
            prefix = "User:" if msg.role == "user" else "Assistant:"
            context_parts.append(f"{prefix} {msg.content}")
            if msg.reasoning:
                context_parts.append(f"  [Reasoning: {msg.reasoning}]")
        
        return "\n".join(context_parts)
    
    def clear(self) -> None:
        """Clear memory."""
        self.messages = []
    
    def get_messages(self) -> List[Message]:
        """Get all messages."""
        return self.messages
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps([asdict(msg) for msg in self.messages], indent=2)

class LongTermMemory:
    """Long-term memory stored in vector DB."""
    
    def __init__(self, vector_db: VectorDB):
        """Initialize long-term memory with vector DB."""
        self.vector_db = vector_db
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.conversation_summaries: List[Dict[str, Any]] = []
    
    def create_conversation_summary(self, 
                                   user_id: str,
                                   messages: List[Message],
                                   summary: str) -> None:
        """
        Store a conversation summary in long-term memory.
        
        Args:
            user_id: User identifier
            messages: Messages in the conversation
            summary: Summary of the conversation
        """
        # Create metadata for the summary
        metadata = {
            "user_id": user_id,
            "type": "conversation_summary",
            "message_count": len(messages),
            "timestamp": datetime.now().isoformat(),
            "first_turn": messages[0].content[:100] if messages else "",
            "last_turn": messages[-1].content[:100] if messages else ""
        }
        
        # Store in vector DB
        doc_id = f"conversation_{user_id}_{datetime.now().timestamp()}"
        self.vector_db.collection.add(
            ids=[doc_id],
            documents=[summary],
            metadatas=[metadata],
            embeddings=[self.vector_db.embedding_model.encode(summary).tolist()]
        )
        
        self.conversation_summaries.append({
            "doc_id": doc_id,
            "user_id": user_id,
            "summary": summary,
            "metadata": metadata
        })
        
        logger.info(f"Stored conversation summary for user {user_id}")
    
    def retrieve_user_context(self, user_id: str, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve past context for a user.
        
        Args:
            user_id: User identifier
            query: Current query to retrieve relevant past context
            top_k: Number of past interactions to retrieve
        
        Returns:
            List of relevant past context
        """
        # Search for relevant past conversations
        results = self.vector_db.retrieve(
            query=query,
            top_k=top_k,
            metadata_filter={"user_id": {"$eq": user_id}}
        )
        
        context = []
        for result in results:
            context.append(result["text"])
        
        logger.debug(f"Retrieved {len(context)} past contexts for user {user_id}")
        return context
    
    def update_user_profile(self, user_id: str, profile_info: Dict[str, Any]) -> None:
        """Update user profile information."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {}
        
        self.user_profiles[user_id].update(profile_info)
        logger.debug(f"Updated profile for user {user_id}")
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile."""
        return self.user_profiles.get(user_id, {})

class ConversationManager:
    """Manage both short-term and long-term memory."""
    
    def __init__(self, user_id: str = None, vector_db: VectorDB = None):
        """Initialize conversation manager."""
        self.user_id = user_id or "default_user"
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory(vector_db) if vector_db else None
        self.turn_count = 0
    
    def add_user_message(self, content: str) -> None:
        """Add user message."""
        self.short_term.add_message("user", content)
        self.turn_count += 1
    
    def add_assistant_message(self, content: str, reasoning: str = None, tool_calls: List[str] = None) -> None:
        """Add assistant message."""
        self.short_term.add_message("assistant", content, reasoning, tool_calls)
    
    def get_context(self) -> str:
        """Get current conversation context."""
        return self.short_term.get_context()
    
    def get_relevant_past_context(self, query: str) -> Optional[str]:
        """Get relevant past context using long-term memory."""
        if not self.long_term:
            return None
        
        past_contexts = self.long_term.retrieve_user_context(self.user_id, query)
        if past_contexts:
            return "\n".join(past_contexts)
        return None
    
    def summarize_and_store(self, summary: str) -> None:
        """Summarize current conversation and store in long-term memory."""
        if self.long_term:
            self.long_term.create_conversation_summary(
                self.user_id,
                self.short_term.get_messages(),
                summary
            )
            logger.info(f"Summarized conversation for user {self.user_id}")
    
    def reset(self) -> None:
        """Reset for new conversation."""
        self.short_term.clear()
        self.turn_count = 0
