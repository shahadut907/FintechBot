"""
Chat API endpoint handling all conversational requests.
This module implements the chat functionality with language detection and translation.
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from src.backend.services.rag_service import RAGService
from src.backend.services.language_utils import detect_language
from src.config.roles import Role
from src.backend.api.endpoints import get_current_user

# Initialize router
router = APIRouter()
logger = logging.getLogger(__name__)

# Define models using Pydantic for validation
class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    role: Optional[str] = None
    user_name: Optional[str] = None
    session_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None

class User(BaseModel):
    """Simple user model"""
    id: str
    role: Role
    
    class Config:
        arbitrary_types_allowed = True

@router.post("/chat")
async def chat(request: ChatRequest, user: Dict[str, Any] = Depends(get_current_user)):
    """Process a chat request and return a response with language support"""
    
    # Initialize or get RAG service
    rag_service = RAGService()
    
    # Process the request
    try:
        # Get user role for RBAC
        if request.role:
            try:
                user_role = Role(request.role)
            except ValueError:
                user_role = Role(user["role"])
        else:
            user_role = Role(user["role"])
        
        # Process the message using RAG with language detection and translation
        response = rag_service.text_generation(
            query=request.message,
            user_role=user_role,
            conversation_id=request.session_id or "default",
            conversation_history=request.conversation_history or []
        )
        
        # Return the response
        return {
            "response": response["text"],
            "session_id": request.session_id,
            "sources": response.get("metadata", {}).get("sources", []),
            "documents_found": response.get("metadata", {}).get("document_count", 0),
            "processing_time": response.get("metadata", {}).get("total_time", "0s"),
            "success": True
        }
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request"
        )
