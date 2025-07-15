"""
FIXED: Enhanced Conversation Memory with Better Query Understanding
Fixes: Context building, identity queries, backend integration, data persistence
"""
import asyncio
import json
import uuid
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading

# Database imports with fallback handling
try:
    import asyncpg
    DB_AVAILABLE = True
except ImportError:
    print("⚠️  asyncpg not available - using in-memory fallback")
    asyncpg = None
    DB_AVAILABLE = False

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    print("⚠️  redis not available - using in-memory fallback")
    redis = None
    REDIS_AVAILABLE = False

# Define the required classes locally to avoid circular import
class MessageType(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ConversationMode(Enum):
    CASUAL = "casual"
    BUSINESS = "business"
    MIXED = "mixed"

@dataclass
class ConversationMessage:
    id: str
    session_id: str
    user_id: str
    message_type: MessageType
    content: str
    timestamp: datetime
    language: str = "english"
    mode: ConversationMode = ConversationMode.BUSINESS
    metadata: Dict[str, Any] = None

@dataclass
class Message:
    """Backward compatibility class"""
    message_id: str
    session_id: str
    user_id: str
    user_role: str
    message_type: MessageType
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ConversationContext:
    """FIXED: Enhanced conversation context with user awareness"""
    session_id: str
    user_id: str
    user_name: str = None
    user_role: str = None
    user_department: str = None
    current_topic: Optional[str] = None
    mentioned_entities: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    conversation_summary: str = None
    identity_confirmed: bool = False
    
    def __post_init__(self):
        if self.mentioned_entities is None:
            self.mentioned_entities = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

class PersistentConversationMemory:
    """
    FIXED: Enhanced conversation memory with better query understanding
    """
    
    def __init__(self, db_pool: 'asyncpg.Pool' = None, redis_client: 'redis.Redis' = None, max_history_length: int = 20, **kwargs):
        self.db_pool = db_pool
        self.redis = redis_client
        self.memory_ttl = kwargs.get('memory_ttl', 3600)  # 1 hour in Redis
        self.max_history_length = max_history_length  # For backward compatibility
        self.context_window_minutes = kwargs.get('context_window_minutes', 60)
        self.context_window = timedelta(minutes=self.context_window_minutes)
        
        # In-memory fallback when database is not available
        self.in_memory_sessions = {}
        self.in_memory_messages = {}
        self.in_memory_contexts = {}
        self._user_sessions = {}
        
        # FIXED: Enhanced user context tracking
        self._user_profiles = {}  # Store user information for better context
        
        # Thread safety for in-memory operations
        self._lock = threading.RLock()
        
        # FIXED: Enhanced pattern matching for better query understanding
        self._setup_enhanced_patterns()
        
        print(f"🧠 FIXED: Enhanced Conversation Memory initialized:")
        print(f"   Database: {'✅' if db_pool else '❌'}")
        print(f"   Redis: {'✅' if redis_client else '❌'}")
        print(f"   In-memory fallback: ✅")
        print(f"   Enhanced query understanding: ✅")
    
    def _setup_enhanced_patterns(self):
        """FIXED: Enhanced pattern recognition for better query understanding"""
        
        # FIXED: Identity query patterns - these should NOT use RAG
        self.identity_patterns = [
            "what is my name", "who am i", "my name is", "what's my name",
            "tell me my name", "do you know my name", "who is speaking",
            "what am i called", "identify me", "my identity"
        ]
        
        # FIXED: User context query patterns - these need user context
        self.user_context_patterns = [
            "my role", "my department", "my position", "what do i do",
            "my job", "my responsibilities", "what is my role",
            "where do i work", "my team", "my access level"
        ]
        
        # FIXED: Conversation history patterns - these need conversation context
        self.history_patterns = [
            "what did we discuss", "what was my last question", "continue our conversation",
            "from our previous chat", "as we talked about", "you mentioned earlier",
            "going back to", "referring to our discussion"
        ]
        
        self.language_patterns = {
            "bangla": [
                "bangla", "বাংলা", "বাংলায়", "kemon acho", "ki khobor", 
                "tumi", "ami", "tomra", "amra", "kemon", "kichu", "kotha", 
                "bolo", "bolba", "cacchi", "jante", "somporke"
            ],
            "english": [
                "english", "speak english", "in english", "switch to english"
            ]
        }
        
        self.casual_indicators = [
            "how are you", "what's up", "hello", "hi", "hey",
            "kemon acho", "ki khobor", "ki obostha", "kemon achen"
        ]
        
        self.topic_keywords = {
            "financial": ["revenue", "profit", "budget", "cost", "expense", "Q1", "Q2", "Q3", "Q4"],
            "hr": ["employee", "staff", "hire", "salary", "performance", "attendance"],
            "marketing": ["campaign", "marketing", "customer", "conversion", "ads"],
            "engineering": ["technical", "system", "development", "code", "infrastructure"]
        }
        
        # FIXED: Add follow-up indicators
        self.follow_up_indicators = [
            "also", "and what about", "how about", "what about", "additionally",
            "furthermore", "besides that", "in addition", "moreover", "plus"
        ]
    
    # FIXED: Enhanced session generation with user context
    def generate_session_id(self, user_id: str, user_name: str = None, user_role: str = None, user_department: str = None) -> str:
        """
        FIXED: Generate session with enhanced user context tracking
        """
        with self._lock:
            session_id = f"session_{user_id}_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
            
            # FIXED: Store comprehensive user context
            user_context = {
                "user_id": user_id,
                "user_name": user_name,
                "user_role": user_role,
                "user_department": user_department,
                "session_start": datetime.now()
            }
            
            # Store user profile for context building
            self._user_profiles[user_id] = user_context
            
            # Store in memory immediately for backward compatibility
            self.in_memory_sessions[session_id] = {
                "user_id": user_id,
                "organization_id": "default",
                "session_name": f"Chat {datetime.now().strftime('%m/%d %H:%M')}",
                "is_active": True,
                "created_at": datetime.now(),
                "current_topic": None,
                "mentioned_entities": [],
                "preferred_language": "english",
                "conversation_mode": "business",
                "casual_interactions": 0,
                "business_interactions": 0,
                **user_context
            }
            
            # FIXED: Initialize enhanced context with user information
            self.in_memory_contexts[session_id] = ConversationContext(
                session_id=session_id,
                user_id=user_id,
                user_name=user_name,
                user_role=user_role,
                user_department=user_department,
                identity_confirmed=True
            )
            
            # Track user sessions
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = []
            self._user_sessions[user_id].append(session_id)
            
            # If database is available, schedule async creation
            if self.db_pool:
                try:
                    asyncio.create_task(self._async_create_session(session_id, user_id, "default", user_context))
                except RuntimeError:
                    pass
            
            return session_id
    
    async def _async_create_session(self, session_id: str, user_id: str, organization_id: str, user_context: Dict):
        """FIXED: Async helper to create session with enhanced context"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO conversation_sessions 
                    (id, user_id, organization_id, session_name, is_active, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (id) DO NOTHING
                """, session_id, user_id, organization_id, 
                    f"Chat {datetime.now().strftime('%m/%d %H:%M')}", True, 
                    json.dumps(user_context))
        except Exception as e:
            print(f"⚠️  Failed to create session in database: {e}")
    
    # FIXED: Enhanced message addition with better context
    def add_message(self, session_id: str, user_id: str, user_role: str, 
                   message_type: MessageType, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        FIXED: Add message with enhanced context and query classification
        """
        with self._lock:
            message_id = f"msg_{uuid.uuid4().hex[:12]}"
            
            # FIXED: Classify the query type for better handling
            query_classification = self._classify_query(content)
            
            # Enhanced metadata with query classification
            enhanced_metadata = metadata or {}
            enhanced_metadata.update({
                "query_type": query_classification["type"],
                "requires_context": query_classification["requires_context"],
                "is_identity_query": query_classification["is_identity"],
                "is_follow_up": query_classification["is_follow_up"]
            })
            
            # Create message object
            message = Message(
                message_id=message_id,
                session_id=session_id,
                user_id=user_id,
                user_role=user_role,
                message_type=message_type,
                content=content,
                timestamp=datetime.now(),
                metadata=enhanced_metadata
            )
            
            # Store in memory
            if session_id not in self.in_memory_messages:
                self.in_memory_messages[session_id] = []
            
            self.in_memory_messages[session_id].append({
                'id': message_id,
                'session_id': session_id,
                'user_id': user_id,
                'user_role': user_role,
                'message_type': message_type.value,
                'content': content,
                'detected_language': self.detect_language(content),
                'conversation_mode': self.detect_conversation_mode(content).value,
                'sources': enhanced_metadata.get('sources', []),
                'created_at': datetime.now(),
                'metadata': enhanced_metadata,
                'query_classification': query_classification
            })
            
            # Keep only recent messages in memory
            if len(self.in_memory_messages[session_id]) > self.max_history_length:
                self.in_memory_messages[session_id] = self.in_memory_messages[session_id][-self.max_history_length:]
            
            # FIXED: Update context with better understanding
            if message_type == MessageType.USER:
                self._update_context_sync(session_id, content, message_type, query_classification)
            
            # If database is available, schedule async storage
            if self.db_pool and message_type in [MessageType.USER, MessageType.ASSISTANT]:
                try:
                    asyncio.create_task(self._async_add_message(
                        session_id, user_id, user_role, message_type, content, enhanced_metadata
                    ))
                except RuntimeError:
                    pass
            
            return message_id
    
    def _classify_query(self, content: str) -> Dict[str, Any]:
        """FIXED: Classify query type for better handling"""
        content_lower = content.lower().strip()
        
        classification = {
            "type": "general",
            "requires_context": True,
            "is_identity": False,
            "is_follow_up": False,
            "confidence": 0.5
        }
        
        # FIXED: Check for identity queries
        if any(pattern in content_lower for pattern in self.identity_patterns):
            classification.update({
                "type": "identity",
                "requires_context": False,  # Should not use RAG
                "is_identity": True,
                "confidence": 0.9
            })
            return classification
        
        # FIXED: Check for user context queries
        if any(pattern in content_lower for pattern in self.user_context_patterns):
            classification.update({
                "type": "user_context",
                "requires_context": True,
                "confidence": 0.8
            })
            return classification
        
        # FIXED: Check for conversation history queries
        if any(pattern in content_lower for pattern in self.history_patterns):
            classification.update({
                "type": "conversation_history",
                "requires_context": True,
                "is_follow_up": True,
                "confidence": 0.9
            })
            return classification
        
        # FIXED: Check for follow-up queries
        if any(indicator in content_lower for indicator in self.follow_up_indicators):
            classification.update({
                "type": "follow_up",
                "requires_context": True,
                "is_follow_up": True,
                "confidence": 0.7
            })
            return classification
        
        # FIXED: Check for casual conversation
        if self.detect_conversation_mode(content) == ConversationMode.CASUAL:
            classification.update({
                "type": "casual",
                "requires_context": False,
                "confidence": 0.8
            })
            return classification
        
        return classification
    
    async def _async_add_message(self, session_id: str, user_id: str, user_role: str,
                                message_type: MessageType, content: str, metadata: Dict = None):
        """FIXED: Async helper with enhanced metadata storage"""
        try:
            detected_language = self.detect_language(content)
            conversation_mode = self.detect_conversation_mode(content)
            
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO conversation_messages 
                    (id, session_id, user_id, organization_id, message_type, content, 
                     detected_language, conversation_mode, sources, processing_time_ms, 
                     documents_found, compliance_level, access_denied, success, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    ON CONFLICT (id) DO NOTHING
                """, 
                    f"msg_{uuid.uuid4().hex[:12]}", session_id, user_id, "default",
                    message_type.value, content, detected_language, conversation_mode.value,
                    json.dumps(metadata.get('sources', [])) if metadata else '[]',
                    metadata.get('processing_time_ms', 0) if metadata else 0,
                    metadata.get('documents_found', 0) if metadata else 0,
                    metadata.get('compliance_level', 'LOW') if metadata else 'LOW',
                    metadata.get('access_denied', False) if metadata else False,
                    metadata.get('success', True) if metadata else True,
                    json.dumps(metadata or {})
                )
        except Exception as e:
            print(f"⚠️  Failed to store message in database: {e}")
    
    # FIXED: Enhanced contextual query building
    def build_contextual_query(self, current_query: str, session_id: str) -> Tuple[str, Dict[str, Any]]:
        """
        FIXED: Build contextual query with better understanding and user awareness
        """
        with self._lock:
            # FIXED: Classify the current query
            query_classification = self._classify_query(current_query)
            
            # Get recent conversation history
            recent_messages = self._get_recent_messages_sync(session_id)
            
            # Get user context
            session_data = self.in_memory_sessions.get(session_id, {})
            context_data = self.in_memory_contexts.get(session_id)
            
            enhanced_query = current_query
            context_metadata = {
                "is_follow_up": len(recent_messages) > 1,
                "conversation_length": len(recent_messages),
                "context_included": False,
                "conversation_mode": self.detect_conversation_mode(current_query).value,
                "language": self.detect_language(current_query),
                "query_type": query_classification["type"],
                "requires_rag": query_classification["requires_context"] and not query_classification["is_identity"]
            }
            
            # FIXED: Handle identity queries - don't use RAG, return user info
            if query_classification["is_identity"]:
                context_metadata["requires_rag"] = False
                context_metadata["user_info_needed"] = True
                
                # Add user context to metadata for direct response
                if context_data:
                    context_metadata["user_name"] = context_data.user_name
                    context_metadata["user_role"] = context_data.user_role
                    context_metadata["user_department"] = context_data.user_department
                
                return current_query, context_metadata
            
            # FIXED: Handle casual queries - minimal context
            if query_classification["type"] == "casual":
                return current_query, context_metadata
            
            # FIXED: Enhanced context building for business queries
            if query_classification["requires_context"] and recent_messages:
                context_parts = []
                
                # Add user context if available
                if context_data and context_data.user_name:
                    user_info = f"User: {context_data.user_name}"
                    if context_data.user_role:
                        user_info += f" ({context_data.user_role}"
                        if context_data.user_department:
                            user_info += f" in {context_data.user_department}"
                        user_info += ")"
                    context_parts.append(user_info)
                
                # Add conversation history for follow-up queries
                if query_classification["is_follow_up"] or len(current_query) < 20 or any(word in current_query.lower() for word in ["previous", "you said", "you mentioned", "earlier", "above", "before"]):
                    user_questions = []
                    assistant_responses = []
                    
                    for msg in recent_messages[-4:]:  # Last 4 messages for context
                        if msg.get('message_type') == 'user':
                            user_questions.append(msg['content'])
                        elif msg.get('message_type') == 'assistant':
                            assistant_responses.append(msg['content'])
                    
                    if user_questions:
                        context_parts.append(f"Previous questions: {' | '.join(user_questions)}")
                    
                    if assistant_responses:
                        # Include key information from previous responses
                        summaries = []
                        for resp in assistant_responses:
                            # Extract first sentence or first 100 chars
                            summary = resp.split('.')[0][:100] + "..." if len(resp) > 100 else resp
                            summaries.append(summary)
                        
                        if summaries:
                            context_parts.append(f"Previous answers contained: {' | '.join(summaries)}")
                
                # Always add current topic for context
                if context_data and context_data.current_topic:
                    context_parts.append(f"Current topic: {context_data.current_topic}")
                
                # Add extracted entities for context
                if context_data and context_data.mentioned_entities:
                    key_entities = context_data.mentioned_entities[-5:]  # Last 5 entities
                    context_parts.append(f"Key entities: {', '.join(key_entities)}")
                
                if context_parts:
                    context_str = "\n".join(context_parts)
                    # Avoid long queries, use a more compact format for longer conversations
                    if len(context_str) > 500:
                        enhanced_query = f"Context: {context_str[:500]}...\nQuery: {current_query}"
                    else:
                        enhanced_query = f"Context: {context_str}\nQuery: {current_query}"
                    context_metadata["context_included"] = True
            
            return enhanced_query, context_metadata
    
    def _get_recent_messages_sync(self, session_id: str) -> List[Dict]:
        """FIXED: Get recent messages within context window with better filtering"""
        if session_id not in self.in_memory_messages:
            return []
        
        now = datetime.now()
        cutoff = now - self.context_window
        
        messages = self.in_memory_messages[session_id]
        recent = []
        
        for msg in messages:
            msg_time = msg.get('created_at')
            if isinstance(msg_time, str):
                try:
                    msg_time = datetime.fromisoformat(msg_time)
                except:
                    msg_time = datetime.now()
            
            if msg_time >= cutoff:
                recent.append(msg)
        
        return recent
    
    def _update_context_sync(self, session_id: str, content: str, message_type: MessageType, query_classification: Dict):
        """FIXED: Update context with enhanced understanding"""
        if session_id not in self.in_memory_contexts:
            session_data = self.in_memory_sessions.get(session_id, {})
            self.in_memory_contexts[session_id] = ConversationContext(
                session_id=session_id,
                user_id=session_data.get('user_id', 'unknown'),
                user_name=session_data.get('user_name'),
                user_role=session_data.get('user_role'),
                user_department=session_data.get('user_department')
            )
        
        context = self.in_memory_contexts[session_id]
        context.updated_at = datetime.now()
        
        # FIXED: Enhanced entity extraction and topic detection
        if message_type == MessageType.USER:
            # Extract entities with better classification
            entities = self._extract_enhanced_entities(content)
            
            for entity in entities[:3]:  # Limit to 3 entities per message
                if entity not in context.mentioned_entities:
                    context.mentioned_entities.append(entity)
            
            # Keep only recent entities
            if len(context.mentioned_entities) > 20:
                context.mentioned_entities = context.mentioned_entities[-20:]
            
            # FIXED: Better topic detection
            new_topic = self.detect_topic_switch(content, context.current_topic)
            if new_topic and new_topic != context.current_topic:
                context.current_topic = new_topic
                
                # Update session data too
                if session_id in self.in_memory_sessions:
                    self.in_memory_sessions[session_id]['current_topic'] = new_topic
            
            # FIXED: Update conversation summary periodically
            if len(self.in_memory_messages.get(session_id, [])) % 5 == 0:  # Every 5 messages
                context.conversation_summary = self._generate_conversation_summary(session_id)
    
    def _extract_enhanced_entities(self, content: str) -> List[str]:
        """FIXED: Enhanced entity extraction with better classification"""
        entities = []
        words = content.split()
        
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word)
            
            # Financial quarters
            if word_clean.upper() in ["Q1", "Q2", "Q3", "Q4"]:
                entities.append(word_clean.upper())
            # Years
            elif re.match(r'20\d{2}', word_clean):
                entities.append(word_clean)
            # Department names
            elif word_clean.lower() in ['finance', 'engineering', 'marketing', 'hr', 'sales']:
                entities.append(word_clean.lower())
            # General entities (filter out common words)
            elif len(word_clean) > 3 and word_clean.isalpha() and word_clean.lower() not in ['what', 'when', 'where', 'how', 'why', 'the', 'and', 'for', 'with']:
                entities.append(word_clean.lower())
        
        return entities[:10]  # Limit entities
    
    def _generate_conversation_summary(self, session_id: str) -> str:
        """FIXED: Generate conversation summary for better context"""
        messages = self.in_memory_messages.get(session_id, [])
        if len(messages) < 3:
            return None
        
        # Extract key topics and questions
        topics = []
        questions = []
        
        for msg in messages[-10:]:  # Last 10 messages
            if msg.get('message_type') == 'user':
                content = msg['content']
                if '?' in content:
                    questions.append(content.split('?')[0] + '?')
                
                # Extract topic keywords
                for topic, keywords in self.topic_keywords.items():
                    if any(keyword in content.lower() for keyword in keywords):
                        if topic not in topics:
                            topics.append(topic)
        
        summary_parts = []
        if topics:
            summary_parts.append(f"Topics discussed: {', '.join(topics)}")
        if questions:
            summary_parts.append(f"Recent questions about: {', '.join(questions[-2:])}")
        
        return " | ".join(summary_parts) if summary_parts else None
    
    # Keep all existing methods for backward compatibility...
    def get_conversation_history(self, session_id: str, limit: Optional[int] = None) -> List['Message']:
        """Get conversation history (backward compatibility)"""
        with self._lock:
            if session_id not in self.in_memory_messages:
                return []
            
            messages = self.in_memory_messages[session_id]
            
            if limit:
                messages = messages[-limit:]
            
            result = []
            for msg in messages:
                try:
                    message_obj = Message(
                        message_id=msg['id'],
                        session_id=msg['session_id'],
                        user_id=msg['user_id'],
                        user_role=msg.get('user_role', 'user'),
                        message_type=MessageType(msg['message_type']),
                        content=msg['content'],
                        timestamp=msg['created_at'] if isinstance(msg['created_at'], datetime) 
                                 else datetime.now(),
                        metadata=msg.get('metadata', {})
                    )
                    result.append(message_obj)
                except Exception as e:
                    print(f"⚠️  Error converting message: {e}")
                    continue
            
            return result
    
    def get_context(self, session_id: str) -> Optional['ConversationContext']:
        """Get conversation context (backward compatibility)"""
        with self._lock:
            return self.in_memory_contexts.get(session_id)
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a conversation session (backward compatibility)"""
        with self._lock:
            try:
                # Remove from memory
                if session_id in self.in_memory_messages:
                    del self.in_memory_messages[session_id]
                
                if session_id in self.in_memory_contexts:
                    del self.in_memory_contexts[session_id]
                
                if session_id in self.in_memory_sessions:
                    del self.in_memory_sessions[session_id]
                
                # Remove from user sessions
                for user_id, sessions in self._user_sessions.items():
                    if session_id in sessions:
                        sessions.remove(session_id)
                
                # If database available, schedule async clear
                if self.db_pool:
                    try:
                        asyncio.create_task(self._async_clear_session(session_id))
                    except RuntimeError:
                        pass
                
                return True
            except Exception as e:
                print(f"⚠️  Error clearing session {session_id}: {e}")
                return False
    
    async def _async_clear_session(self, session_id: str):
        """Async helper to clear session in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE conversation_sessions 
                    SET is_active = FALSE, updated_at = NOW()
                    WHERE id = $1
                """, session_id)
            
            # Clear Redis cache
            if self.redis:
                await self.redis.delete(f"session:{session_id}")
                await self.redis.delete(f"messages:{session_id}")
        except Exception as e:
            print(f"⚠️  Failed to clear session in database: {e}")
    
    def get_user_sessions(self, user_id: str) -> List[str]:
        """Get all sessions for a user (backward compatibility)"""
        with self._lock:
            return self._user_sessions.get(user_id, []).copy()
    
    # Keep all existing methods...
    def detect_language(self, content: str) -> str:
        """Language detection"""
        content_lower = content.lower()
        
        bangla_score = sum(1 for pattern in self.language_patterns["bangla"] 
                          if pattern in content_lower)
        english_score = sum(1 for pattern in self.language_patterns["english"] 
                           if pattern in content_lower)
        
        if re.search(r'[আ-ঝ]', content):
            bangla_score += 3
        
        return "bangla" if bangla_score > english_score else "english"
    
    def detect_conversation_mode(self, content: str) -> ConversationMode:
        """Conversation mode detection"""
        content_lower = content.lower()
        
        casual_score = sum(1 for indicator in self.casual_indicators 
                          if indicator in content_lower)
        
        business_score = 0
        for topic, keywords in self.topic_keywords.items():
            business_score += sum(1 for keyword in keywords 
                                if keyword in content_lower)
        
        if casual_score > 0 and business_score == 0:
            return ConversationMode.CASUAL
        elif business_score > 0 and casual_score == 0:
            return ConversationMode.BUSINESS
        else:
            return ConversationMode.MIXED
    
    def detect_topic_switch(self, content: str, current_topic: str = None) -> Optional[str]:
        """Topic switching detection"""
        content_lower = content.lower()
        
        # Check for explicit topic switching
        topic_switch_phrases = [
            "let's talk about", "switch to", "move to", "discuss", 
            "jante cacchi", "somporke", "about", "regarding"
        ]
        
        for phrase in topic_switch_phrases:
            if phrase in content_lower:
                for topic, keywords in self.topic_keywords.items():
                    if any(keyword in content_lower for keyword in keywords):
                        if topic != current_topic:
                            return topic
        
        # Regular topic detection
        detected_topics = []
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                detected_topics.append(topic)
        
        if detected_topics:
            new_topic = detected_topics[0]
            if new_topic != current_topic:
                return new_topic
        
        return current_topic

# Export classes for easy importing
__all__ = [
    'MessageType',
    'ConversationMode', 
    'ConversationMessage',
    'Message',
    'ConversationContext',
    'PersistentConversationMemory',
    'ConversationMemory'
]

# Backward compatibility
ConversationMemory = PersistentConversationMemory