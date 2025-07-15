"""
GPU-Optimized RAG Service for Maximum Performance
Addresses all performance bottlenecks with GPU acceleration
"""
import requests
import json
import torch
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import time
import asyncio
import threading
import logging
from functools import lru_cache
import gc
import re
import os
from pathlib import Path
import hashlib
from enum import Enum

from src.config.settings import settings
from src.config.roles import Role, get_accessible_documents, DOCUMENT_CATEGORIES, can_access_document
from src.backend.services.vector_store import VectorStore
from src.backend.utils.document_processor import DocumentProcessor
from src.backend.services.language_utils import detect_language, translate_text, translate_content_blocks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a logger for this module
logger = logging.getLogger(__name__)

class RAGService:
    """
    GPU-Optimized RAG System with Performance Enhancements
    Fixes: GPU utilization, caching, document reloading, batch processing
    """
    
    _instance = None
    _initialized = False
    _document_cache = {}
    _embedding_cache = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the RAG service with vector store and caching"""
        print("🚀 Initializing GPU-Optimized RAG Service...")
        
        try:
            print("🔧 Setting up GPU environment...")
            # Check GPU capability for optimizations
            self.cuda_available = False  # Default to False
            try:
                import torch
                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0)
                    memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    print(f"✅ CUDA available: {device_name}")
                    print(f"📊 GPU Memory: {memory:.1f} GB")
                    self.cuda_available = True  # Set to True if CUDA is available
                else:
                    print("⚠️ CUDA not available, using CPU")
            except Exception as e:
                print(f"⚠️ Error checking CUDA: {e}")
                
            # Initialize vector store for document retrieval
            from src.backend.services.vector_store import VectorStore
            self.vector_store = VectorStore()
            
            # Initialize document processor
            from src.backend.utils.document_processor import DocumentProcessor
            self.document_processor = DocumentProcessor()
            
            # Initialize response caching
            self._response_cache = {}
            
            # Document refresh settings
            self._last_document_check = 0
            self._document_check_interval = 300  # Check for doc changes every 5 minutes
            self._documents_loaded = False
            self._last_document_status = None
            
            # Ollama configuration
            self.ollama_url = settings.ollama_base_url
            self.model_name = settings.ollama_model
            self.max_tokens = min(settings.max_tokens, 512)
            
            # Set up performance optimizations
            self._setup_performance_optimizations()
            
            # CRITICAL: Pre-load model and warm caches
            self._initialize_gpu_pipeline()
            
            # Check document status without triggering reload
            self._check_document_status()
            
            self._initialized = True
            print(f"✅ GPU-Optimized RAG Service ready with model: {self.model_name}")
            
        except Exception as e:
            print(f"❌ Failed to initialize optimized RAG Service: {e}")
            logger.error(f"RAG Service initialization error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _setup_gpu_environment(self):
        """Setup GPU environment for optimal performance"""
        print("🔧 Setting up GPU environment...")
        
        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.device = torch.device("cuda")
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Optimize CUDA settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            gc.collect()
            
        else:
            self.device = torch.device("cpu")
            print("⚠️ CUDA not available, optimizing CPU performance...")
            
            # CPU optimizations
            torch.set_num_threads(min(8, torch.get_num_threads()))

    def _setup_performance_optimizations(self):
        """Setup caching and performance tracking"""
        self._performance_stats = {
            "total_requests": 0,
            "avg_response_time": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "gpu_acceleration": self.cuda_available,
            "document_reloads": 0,
            "embedding_batch_size": 32 if self.cuda_available else 8
        }
        
        # Response cache for repeated queries
        self._response_cache = {}
        self._max_cache_size = 1000
        
        # Embedding cache for queries
        self._query_embedding_cache = {}
        
        # Document status tracking
        self._last_document_check = 0
        self._document_check_interval = 300  # 5 minutes

    def _initialize_gpu_pipeline(self):
        """Initialize and warm up GPU pipeline components"""
        print("🔥 Warming up GPU pipeline...")
        
        try:
            # Keep Ollama model warm
            self._keep_model_warm()
            
            # Pre-warm embedding model if using GPU
            if self.cuda_available and hasattr(self.vector_store, 'embeddings_model'):
                print("🔥 Pre-warming embedding model on GPU...")
                test_texts = [
                    "test query for warming up",
                    "financial data sample",
                    "engineering documentation",
                    "hr policy information"
                ]
                
                # Batch embedding test
                start_time = time.perf_counter()
                self.vector_store._generate_batch_embeddings(test_texts)
                warmup_time = time.perf_counter() - start_time
                
                print(f"✅ Embedding model warmed up in {warmup_time:.2f}s")
                
        except Exception as e:
            print(f"⚠️ Pipeline warmup error: {e}")

    def _check_document_status(self):
        """Check document status without triggering unnecessary reloads.
        Also ensure the vector store is synchronised with live source files to avoid context leakage."""
        current_time = time.time()
        
        # Only check periodically, not on every request
        if current_time - self._last_document_check < self._document_check_interval:
            return
        
        try:
            # ----- NEW: Synchronise vector store with live files ----- #
            self._sync_documents("src/data/raw")
            # -------------------------------------------------------- #
            
            stats = self.vector_store.get_collection_stats()
            total_docs = stats.get("total_documents", 0)
            
            if total_docs == 0:
                print("⚠️ No documents in vector store - consider loading documents")
            else:
                print(f"📊 Vector store status: {total_docs} documents ready")
                departments = stats.get("departments", {})
                print(f"📁 By department: {departments}")
            
            self._last_document_check = current_time
            
        except Exception as e:
            print(f"⚠️ Document status check failed: {e}")

    def is_ready(self) -> bool:
        """Check if RAG service is ready for high-performance processing"""
        return (self._initialized and 
                hasattr(self, 'vector_store') and 
                self.vector_store is not None and
                hasattr(self, 'ollama_url'))

    def _keep_model_warm(self):
        """Keep Ollama model loaded and warm"""
        try:
            warmup_response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": "Hello",
                    "stream": False,
                    "options": {
                        "num_predict": 5,
                        "temperature": 0.1,
                        "keep_alive": -1,  # Keep in memory
                        "num_gpu": -1 if self.cuda_available else 0
                    }
                },
                timeout=30
            )
            
            if warmup_response.status_code == 200:
                print("✅ Ollama model warmed up on GPU")
                self._start_keepalive_thread()
            
        except Exception as e:
            print(f"❌ Model warmup error: {e}")

    def _start_keepalive_thread(self):
        """Background thread to keep model active"""
        def keepalive():
            while True:
                try:
                    time.sleep(300)  # Every 5 minutes
                    requests.post(
                        f"{self.ollama_url}/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": "ping",
                            "stream": False,
                            "options": {
                                "num_predict": 1,
                                "keep_alive": -1,
                                "num_gpu": -1 if self.cuda_available else 0
                            }
                        },
                        timeout=5
                    )
                except:
                    pass
        
        thread = threading.Thread(target=keepalive, daemon=True)
        thread.start()

    @lru_cache(maxsize=500)
    def _get_cached_response(self, query_hash: str, user_role: str) -> Optional[Dict]:
        """Cache responses for identical queries from same role"""
        cache_key = f"{query_hash}_{user_role}"
        return self._response_cache.get(cache_key)

    def _cache_response(self, query_hash: str, user_role: str, response: Dict):
        """Cache response with size management"""
        cache_key = f"{query_hash}_{user_role}"
        
        if len(self._response_cache) >= self._max_cache_size:
            # Remove oldest entries
            oldest_keys = list(self._response_cache.keys())[:100]
            for key in oldest_keys:
                del self._response_cache[key]
        
        self._response_cache[cache_key] = response

    def process_query(self, query: str, user_role: Role, user_name: str = "User") -> Dict[str, Any]:
        """
        GPU-Optimized RAG pipeline with intelligent caching
        """
        pipeline_start = time.perf_counter()
        
        # Generate query hash for caching
        query_hash = str(hash(query.lower().strip()))
        
        try:
            # Quick greeting/casual small-talk handler – skip heavy RAG pipeline
            normalized_q = query.lower().strip()
            greetings = ["hi", "hello", "hey", "how are you", "good morning", "good afternoon", "good evening"]
            identity_qs = [
                "who are you", "what is your name", "what's your name", "identify yourself", 
                "tell me about yourself", "what are you", "who you are"
            ]

            # Fast-path greeting with strict word-boundary matching to prevent false positives (e.g. "hi" inside "this")
            greeting_detected = False
            if len(normalized_q.split()) <= 6:
                for g in greetings:
                    if " " in g:  # multi-word greeting – require exact match
                        if normalized_q == g:
                            greeting_detected = True
                            break
                    else:
                        if re.search(rf"\b{re.escape(g)}\b", normalized_q):
                            greeting_detected = True
                            break

            if greeting_detected:
                greeting_response = {
                    "success": True,
                    "response": f"👋 Hello {user_name}! I'm FinSolve AI, your in-house assistant 😊. How can I assist you today?",
                    "sources": [],
                    "user_role": user_role.value,
                    "documents_found": 0,
                    "processing_time": "0.00s",
                    "timestamp": datetime.now().isoformat(),
                    "access_denied": False,
                    "gpu_accelerated": self.cuda_available,
                    "from_cache": False
                }
                return greeting_response

            # Fast-path assistant-identity questions using word-boundary matching as well
            if any(re.search(rf"\b{re.escape(pattern)}\b", normalized_q) for pattern in identity_qs):
                assistant_intro = (
                    f"I'm FinSolve AI 🤖, the in-house assistant for {user_role.value.lower()} staff. "
                    "I'm here to help with data and document queries. How can I assist you today? 😊"
                )
                identity_response = {
                    "success": True,
                    "response": assistant_intro,
                    "sources": [],
                    "user_role": user_role.value,
                    "documents_found": 0,
                    "processing_time": "0.00s",
                    "timestamp": datetime.now().isoformat(),
                    "access_denied": False,
                    "gpu_accelerated": self.cuda_available,
                    "from_cache": False
                }
                return identity_response
            
            # Check cache first
            cached_response = self._get_cached_response(query_hash, user_role.value)
            if cached_response:
                self._performance_stats["cache_hits"] += 1
                cached_response["from_cache"] = True
                cached_response["processing_time"] = "0.01s"
                return cached_response
            
            self._performance_stats["cache_misses"] += 1
            
            # Check if service is ready
            if not self.is_ready():
                return self._create_service_unavailable_response()

            self._performance_stats["total_requests"] += 1
            
            # STEP 1: RBAC check with caching
            rbac_start = time.perf_counter()
            accessible_docs = get_accessible_documents(user_role)
            rbac_time = time.perf_counter() - rbac_start
            
            if settings.debug:
                print(f"🔍 RBAC: {user_name} ({user_role.value}) → {len(accessible_docs)} accessible docs")
            
            if not accessible_docs:
                return self._create_access_denied_response(query, user_role, user_name)
            
            # STEP 2: GPU-Accelerated document search
            search_start = time.perf_counter()
            relevant_docs = self._search_documents_gpu_optimized(
                query=query,
                user_role=user_role,
                accessible_docs=accessible_docs,
                limit=3
            )
            search_time = time.perf_counter() - search_start
            
            if not relevant_docs:
                return self._create_no_results_response(query, user_role, user_name, accessible_docs)
            
            # STEP 3: RBAC verification with batch processing
            context_start = time.perf_counter()
            verified_docs = self._verify_documents_batch(relevant_docs, user_role)
            
            if not verified_docs:
                return self._create_no_results_response(query, user_role, user_name, accessible_docs)
            
            context = self._prepare_optimized_context(verified_docs, user_role)
            context_time = time.perf_counter() - context_start
            
            # STEP 4: GPU-accelerated LLM inference
            llm_start = time.perf_counter()
            ai_response = self._generate_gpu_optimized_response(query, context, user_role, user_name)
            llm_time = time.perf_counter() - llm_start
            
            # STEP 5: Prepare and cache response
            total_time = time.perf_counter() - pipeline_start
            
            response = {
                "success": True,
                "response": ai_response,
                "sources": [doc.get('source', '') for doc in verified_docs],
                "user_role": user_role.value,
                "documents_found": len(verified_docs),
                "processing_time": f"{total_time:.2f}s",
                "timestamp": datetime.now().isoformat(),
                "access_denied": False,
                "gpu_accelerated": self.cuda_available,
                "from_cache": False,
                "performance_breakdown": {
                    "rbac_check": f"{rbac_time:.3f}s",
                    "gpu_search": f"{search_time:.3f}s", 
                    "context_prep": f"{context_time:.3f}s",
                    "gpu_inference": f"{llm_time:.3f}s",
                    "total_pipeline": f"{total_time:.3f}s"
                } if settings.debug else {}
            }
            
            # Cache the response
            self._cache_response(query_hash, user_role.value, response.copy())
            
            # Update performance stats
            self._update_performance_stats(total_time)
            
            if settings.debug:
                print(f"✅ GPU-optimized response: {total_time:.2f}s (GPU: {self.cuda_available})")
            
            return response
            
        except Exception as e:
            total_time = time.perf_counter() - pipeline_start
            print(f"❌ GPU pipeline error: {e}")
            logger.error(f"Query processing error: {e}")
            
            return {
                "success": False,
                "response": "I apologize, but I encountered an error while processing your query. Please try again.",
                "sources": [],
                "user_role": user_role.value,
                "documents_found": 0,
                "processing_time": f"{total_time:.2f}s",
                "timestamp": datetime.now().isoformat(),
                "access_denied": False,
                "gpu_accelerated": self.cuda_available,
                "error": str(e) if settings.debug else None
            }

    def _search_documents_gpu_optimized(self, query: str, user_role: Role, accessible_docs: List[str], limit: int = 3) -> List[Dict[str, Any]]:
        """
        GPU-optimized document search with higher recall for better relevance
        
        Args:
            query: The query to search for
            user_role: User's role for RBAC
            accessible_docs: List of accessible document IDs
            limit: Maximum number of documents to return
            
        Returns:
            List of documents with content and metadata
        """
        try:
            print(f"\n🔍 Searching for documents with query: '{query}'")
            # Use existing search_documents method but increase the limit for better recall
            results = self.vector_store.search_documents(
                query=query, 
                user_role=user_role,
                limit=limit
            )
            
            # Filter and sort results
            filtered_results = []
            for result in results:
                # Calculate similarity
                similarity = result.get("similarity", 0.0)
                
                # Only include results that meet the quality threshold
                if similarity >= 0.4:  # Increased quality threshold
                    filtered_results.append(result)
                    
            # Sort by similarity (highest first)
            filtered_results.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
            
            # Return the top results
            return filtered_results[:limit]
            
        except Exception as e:
            logger.error(f"Error in GPU optimized search: {e}")
            return []

    def _user_can_access_department(self, user_role: Role, doc_department: str) -> bool:
        """Check if user role can access document from specific department"""
        # General/company-wide documents - everyone can access
        if doc_department in ['general', 'company_wide']:
            return True
        
        # Department-specific access
        role_department_map = {
            Role.FINANCE: ['finance'],
            Role.HR: ['hr'],
            Role.MARKETING: ['marketing'], 
            Role.ENGINEERING: ['engineering'],
            Role.C_LEVEL: ['finance', 'hr', 'marketing', 'engineering', 'executive'],  # C-level sees all
            Role.EMPLOYEE: ['general']  # Employees only see general
        }
        
        allowed_departments = role_department_map.get(user_role, ['general'])
        return doc_department in allowed_departments

    def _verify_documents_batch(self, docs: List[Dict], user_role: Role) -> List[Dict]:
        """Batch verification of document access with minimal overhead"""
        verified_docs = []
        rbac_blocks = 0
        
        for doc in docs:
            source = doc.get('source', '')
            if can_access_document(user_role, source):
                verified_docs.append(doc)
            else:
                rbac_blocks += 1
                if settings.debug:
                    print(f"🚫 RBAC: Batch blocked {source}")
        
        if settings.debug and rbac_blocks > 0:
            print(f"🔒 RBAC: Blocked {rbac_blocks} documents in batch")
        
        return verified_docs

    def _prepare_optimized_context(self, docs: List[Dict], user_role: Role) -> str:
        """Prepare context with optimal chunking for GPU processing"""
        if not docs:
            return "No relevant documents found."
        
        context_parts = []
        max_context_length = 2048  # Optimize for GPU memory
        current_length = 0
        
        for i, doc in enumerate(docs[:3], 1):
            content = doc.get('content', '')
            source = doc.get('source', 'Unknown')
            
            # Intelligent truncation based on content importance
            if len(content) > 800:
                # Keep first and last parts for context
                content = content[:600] + "..." + content[-200:]
            
            chunk = f"Document {i} ({source}): {content}"
            
            if current_length + len(chunk) > max_context_length:
                break
            
            context_parts.append(chunk)
            current_length += len(chunk)
        
        return "\n\n".join(context_parts)

    def _generate_gpu_optimized_response(self, query: str, context: str, user_role: Role, user_name: str) -> str:
        """Generate response with GPU-optimized settings"""
        
        role_prompts = {
            Role.EMPLOYEE: "You are a helpful assistant. Answer based on provided documents.",
            Role.FINANCE: "You are a financial assistant focusing on financial data and analysis.",
            Role.HR: "You are an HR assistant focusing on employee and policy information.",
            Role.ENGINEERING: "You are a technical assistant focusing on engineering documentation.",
            Role.MARKETING: "You are a marketing assistant focusing on marketing and customer data.",
            Role.C_LEVEL: "You are an executive assistant with comprehensive company access."
        }
        
        # Add global guidance to make answers concise, conversational, and clarification-aware
        guidance = (
            "\n\nGuidelines:\n"
            "• If the context is insufficient to answer confidently, ask the user a clarifying question instead of guessing.\n"
            "• Keep answers focused on the user's question; avoid introducing unrelated document details.\n"
            "• Be direct, precise, and conversational. Aim for ≤150 words unless more detail is explicitly requested.\n"
            "• When appropriate, include a relevant emoji to convey empathy, positivity, or sentiment.\n"
        )

        system_prompt_with_guidance = role_prompts.get(user_role, role_prompts[Role.EMPLOYEE]) + guidance
        
        # Optimized prompt for GPU processing with guidance
        full_prompt = f"""{system_prompt_with_guidance}

Context: {context}

Question: {query}

First, think about whether you have enough information. If yes, answer directly. Otherwise, ask a clarifying question."""
        
        return self._call_ollama_gpu_optimized(full_prompt)

    def _call_ollama_gpu_optimized(self, prompt: str) -> str:
        """GPU-optimized Ollama inference with performance tuning"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 512,
                        "num_ctx": 4096,
                        "num_gpu": -1 if self.cuda_available else 0,  # Use all GPU layers
                        "num_thread": 4,
                        "repeat_penalty": 1.1,
                        "stop": ["Question:", "Context:", "\n\nUser:"],
                        "keep_alive": -1  # Keep in GPU memory
                    }
                },
                timeout=15  # Shorter timeout for GPU processing
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', '').strip()
                return ai_response if ai_response else "I couldn't generate a proper response."
            else:
                return "I'm experiencing technical difficulties. Please try again."
                
        except requests.exceptions.Timeout:
            return "Response timeout. Please try again."
        except Exception as e:
            logger.error(f"GPU LLM error: {e}")
            return "I encountered an error. Please try again."

    def _update_performance_stats(self, request_time: float):
        """Update performance statistics"""
        if self._performance_stats["total_requests"] > 0:
            self._performance_stats["avg_response_time"] = (
                (self._performance_stats["avg_response_time"] * (self._performance_stats["total_requests"] - 1) + request_time) 
                / self._performance_stats["total_requests"]
            )

    def _create_service_unavailable_response(self) -> Dict[str, Any]:
        """Service unavailable response"""
        return {
            "success": False,
            "response": "System is initializing. Please try again in a moment.",
            "sources": [],
            "user_role": "UNKNOWN",
            "documents_found": 0,
            "processing_time": "0.00s",
            "timestamp": datetime.now().isoformat(),
            "access_denied": False,
            "gpu_accelerated": self.cuda_available
        }

    def _create_access_denied_response(self, query: str, user_role: Role, user_name: str) -> Dict[str, Any]:
        """Access denied response with guidance"""
        return {
            "success": True,
            "response": (
                f"I'm sorry, but I don't have permission to view the documents needed to answer that for a {user_role.value}. "
                "If you believe you should have access, please let me know or contact an administrator."
            ),
            "sources": [],
            "user_role": user_role.value,
            "documents_found": 0,
            "processing_time": "0.01s",
            "timestamp": datetime.now().isoformat(),
            "access_denied": True,
            "gpu_accelerated": self.cuda_available
        }

    def _create_no_results_response(self, query: str, user_role: Role, user_name: str, accessible_docs: List[str]) -> Dict[str, Any]:
        """No results response that maintains FinSolve AI identity"""
        return {
            "success": True,
            "response": (
                f"As FinSolve AI, I don't have specific information about '{query}' in the company database. "
                f"However, I can help you with financial analysis, company policies, employee information, or other business data. "
                f"Would you like me to assist you with a related topic or provide information about what data is available to {user_role.value} users?"
            ),
            "sources": accessible_docs[:3],
            "user_role": user_role.value,
            "documents_found": 0,
            "processing_time": "0.01s",
            "timestamp": datetime.now().isoformat(),
            "access_denied": False,
            "gpu_accelerated": self.cuda_available
        }

    def _fallback_search(self, query: str, user_role: Role, accessible_docs: List[str], limit: int) -> List[Dict[str, Any]]:
        """Fallback search method"""
        try:
            docs = self.vector_store.similarity_search(query, k=limit*2)
            filtered_docs = []
            
            for doc in docs:
                if len(filtered_docs) >= limit:
                    break
                    
                source = getattr(doc, 'metadata', {}).get('source', '')
                if can_access_document(user_role, source):
                    filtered_docs.append({
                        'content': getattr(doc, 'page_content', ''),
                        'source': source,
                        'metadata': getattr(doc, 'metadata', {})
                    })
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Fallback search error: {e}")
            return []

    def get_system_info(self) -> Dict[str, Any]:
        """Get GPU-optimized system information"""
        try:
            ollama_status = self._check_ollama_status()
            
            vector_stats = {}
            if hasattr(self, 'vector_store') and self.vector_store:
                try:
                    vector_stats = self.vector_store.get_collection_stats()
                except:
                    vector_stats = {"total_documents": 0, "status": "unknown"}
            
            gpu_info = {}
            if self.cuda_available:
                gpu_info = {
                    "gpu_name": torch.cuda.get_device_name(),
                    "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
                    "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1e9:.2f} GB",
                    "gpu_memory_cached": f"{torch.cuda.memory_reserved() / 1e9:.2f} GB"
                }
            
            return {
                "ollama_status": ollama_status,
                "vector_store": vector_stats,
                "model": getattr(self, 'model_name', 'unknown'),
                "ollama_url": getattr(self, 'ollama_url', 'unknown'),
                "status": "operational" if ollama_status else "degraded",
                "gpu_acceleration": self.cuda_available,
                "gpu_info": gpu_info,
                "performance_stats": getattr(self, '_performance_stats', {}),
                "cache_stats": {
                    "response_cache_size": len(self._response_cache),
                    "cache_hit_rate": f"{(self._performance_stats.get('cache_hits', 0) / max(self._performance_stats.get('total_requests', 1), 1)):.2%}"
                }
            }
            
        except Exception as e:
            logger.error(f"System info error: {e}")
            return {"status": "error", "error": str(e)}

    def _check_ollama_status(self) -> bool:
        """Check Ollama status"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def load_documents(self, directory_path: str = "src/data/raw") -> Dict[str, Any]:
        """Load documents with performance tracking"""
        print(f"📚 GPU-optimized document loading from: {directory_path}")
        
        start_time = time.perf_counter()
        self._performance_stats["document_reloads"] += 1
        
        try:
            if not hasattr(self, 'document_processor'):
                return {"success": False, "message": "Document processor not available"}
            
            chunks = self.document_processor.process_directory(directory_path)
            
            if not chunks:
                return {"success": False, "message": "No documents found"}
            
            # Clear existing data
            if hasattr(self.vector_store, 'clear_collection'):
                self.vector_store.clear_collection()
            
            # GPU-accelerated document addition
            success = self.vector_store.add_documents(chunks)
            
            load_time = time.perf_counter() - start_time
            
            if success:
                # Clear caches after document reload
                self._response_cache.clear()
                self._query_embedding_cache.clear()
                
                documents = list(set([chunk.metadata['source'] for chunk in chunks]))
                departments = {}
                for chunk in chunks:
                    dept = chunk.metadata.get('department', 'unknown')
                    dept_name = dept.value if hasattr(dept, 'value') else str(dept)
                    departments[dept_name] = departments.get(dept_name, 0) + 1
                
                print(f"✅ GPU-optimized loading: {len(chunks)} chunks in {load_time:.2f}s")
                
                return {
                    "success": True,
                    "message": f"GPU-optimized loading: {len(chunks)} chunks in {load_time:.2f}s",
                    "chunks_processed": len(chunks),
                    "documents": documents,
                    "departments": departments,
                    "load_time": f"{load_time:.2f}s",
                    "gpu_accelerated": self.cuda_available
                }
            else:
                return {"success": False, "message": "Failed to add documents"}
                
        except Exception as e:
            load_time = time.perf_counter() - start_time
            logger.error(f"Document loading error: {e}")
            return {
                "success": False,
                "message": f"Error loading documents: {str(e)}",
                "load_time": f"{load_time:.2f}s"
            }

    # ------------------------------------------------------------------ #
    # New helper to keep vector store in sync with filesystem            #
    # ------------------------------------------------------------------ #
    def _sync_documents(self, directory_path: str = "src/data/raw"):
        """Synchronize vector store with source documents to avoid context leakage"""
        try:
            # Get existing document sources from vector store
            existing_sources = set(self.vector_store.get_unique_sources())
            
            # Get current files in directory
            current_files = set()
            for file_path in Path(directory_path).glob("**/*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.md', '.txt', '.csv', '.pdf', '.docx']:
                    current_files.add(file_path.name)
            
            # Find sources that no longer exist in the filesystem
            deleted_sources = existing_sources - current_files
            
            # Delete documents from vector store if source files are gone
            if deleted_sources:
                print(f"🔄 Removing {len(deleted_sources)} sources no longer in filesystem: {deleted_sources}")
                deleted_count = self.vector_store.delete_documents_by_sources(list(deleted_sources))
                print(f"🗑️ Deleted {deleted_count} document chunks from vector store")
            
            # Check if any new files need to be processed
            new_files = current_files - existing_sources
            if new_files:
                print(f"🔄 Found {len(new_files)} new files to process")
                self.load_documents(directory_path)
                
            # Force collection refresh to ensure consistency
            self.vector_store.collection.get()
                
        except Exception as sync_err:
            print(f"⚠️ Document sync failed: {sync_err}")

    def text_generation(self, query: str, user_role: Role, conversation_id: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Generate text based on a query and conversation history using a RAG (Retrieval Augmented Generation) approach.
        
        Args:
            query: The user query
            user_role: The role of the user for RBAC
            conversation_id: The ID of the conversation for cache lookup
            conversation_history: Previous conversation turns
            
        Returns:
            Dict containing the generated text and metadata
        """
        # Initialize history if None
        if conversation_history is None:
            conversation_history = []
            # Reset language for new conversations
            from src.backend.services.language_utils import reset_session_language
            reset_session_language(conversation_id)
            
        # Check document status and reload if needed
        self._check_document_status()
        
        # DEBUG: Start detailed pipeline logging
        print(f"\n🔍 DEBUG: Starting RAG pipeline for query: '{query}'")
        # Handle both string and enum role values
        role_value = user_role if isinstance(user_role, str) else user_role.value
        print(f"👤 User role: {role_value}, Conversation ID: {conversation_id}")
        print(f"💬 History items: {len(conversation_history)}")
        
        # Force English for standard greetings regardless of language detection
        greetings = ['hey', 'hi', 'hello', 'yo']
        if query.lower().strip() in greetings:
            from src.backend.services.language_utils import set_session_language
            set_session_language(conversation_id, 'en')
            return {
                "text": f"Hello! I'm FinSolve AI, your intelligent assistant. How can I help you today?",
                "metadata": {
                    "query_language": "en",
                    "sources": [],
                    "document_count": 0,
                    "retrieval_time": "0.00s",
                    "llm_time": "0.00s",
                    "total_time": "0.00s",
                    "confidence": 0.95  # High confidence for standard greetings
                }
            }
            
        # Only use cache for identical queries with identical conversation IDs
        # Don't cache complex follow-up queries that depend on conversation context
        cache_key = None
        if len(query) > 10 and not any(marker in query.lower() for marker in ["previous", "last time", "you said", "earlier", "before", "you mentioned"]):
            cache_key = f"{conversation_id}:{query}"
            if cache_key in self._response_cache:
                logger.info(f"Using cached response for query in conversation {conversation_id}")
                print(f"🔄 Using cached response for conversation {conversation_id}")
                return self._response_cache[cache_key]
            
        try:
            # STEP 1: Detect query language with session tracking
            print(f"\n🌐 STEP 1: Detecting language with session awareness...")
            query_lang, confidence = detect_language(query, session_id=conversation_id)
            print(f"🌐 Detected language: {query_lang} (confidence: {confidence:.2f})")
            logger.info(f"Detected query language: {query_lang} (confidence: {confidence:.2f})")
            
            # STEP 2: Translate query to English for better retrieval if not already in English
            print(f"\n🔄 STEP 2: Translating query (if needed)...")
            retrieval_query = query
            if query_lang != 'en' and confidence > 0.5:
                print(f"🔄 Original query: '{query}'")
                retrieval_query = translate_text(query, target_lang='en', source_lang=query_lang)
                print(f"🔄 Translated query: '{retrieval_query}'")
                logger.info(f"Translated query for retrieval: {retrieval_query}")
            else:
                print(f"🔄 No translation needed, using original query")
                
            # STEP 2.3: Special handling for financial queries
            financial_query = False
            financial_terms = ["revenue", "q1", "q2", "q3", "q4", "financial", "profit", "income", "earnings", "sales", "quarter", "fiscal"]
            
            if any(term in retrieval_query.lower() for term in financial_terms):
                print(f"💰 Financial query detected: '{retrieval_query}'")
                financial_query = True
                # Always enhance financial queries to improve document retrieval
                retrieval_query = f"financial data revenue earnings {retrieval_query}"
                
            # STEP 2.5: Process conversation history to improve query understanding
            if conversation_history and len(conversation_history) > 0:
                print(f"\n🧠 STEP 2.5: Enhancing query with conversation context...")
                try:
                    # Create a mock conversation memory service for contextual processing
                    from src.backend.services.conversation_memory import PersistentConversationMemory, MessageType
                    
                    # Create temporary session
                    conv_memory = PersistentConversationMemory()
                    temp_session_id = f"temp_{conversation_id}"
                    temp_user_id = "user"
                    
                    # Add recent history to memory
                    for turn in conversation_history[-5:]:  # Last 5 turns
                        user_msg = turn.get("user", "")
                        assistant_msg = turn.get("assistant", "")
                        if user_msg:
                            conv_memory.add_message(temp_session_id, temp_user_id, str(user_role), 
                                                   MessageType.USER, user_msg)
                        if assistant_msg:
                            conv_memory.add_message(temp_session_id, temp_user_id, str(user_role), 
                                                   MessageType.ASSISTANT, assistant_msg)
                    
                    # Build enhanced query with context
                    enhanced_query, context_meta = conv_memory.build_contextual_query(retrieval_query, temp_session_id)
                    
                    print(f"🧠 Enhanced query: '{enhanced_query[:100]}...' (original: '{retrieval_query[:50]}...')")
                    
                    # Check if we should use RAG or not based on query type
                    if not context_meta.get("requires_rag", True):
                        print(f"🧠 Context suggests skipping RAG for this query type: {context_meta.get('query_type', 'unknown')}")
                        search_results = []
                    else:
                        # Use enhanced query for document retrieval
                        retrieval_query = enhanced_query
                except Exception as e:
                    print(f"⚠️ Context enhancement failed: {e}, using original query")
            
            # STEP 3: Get relevant documents from the vector store
            print(f"\n📚 STEP 3: Retrieving documents...")
            search_start = time.time()
            
            # Increase limit for financial queries to get more context
            limit = 5 if financial_query else 3
            
            search_results = self._search_documents_gpu_optimized(
                query=retrieval_query,  # Use enhanced query for better retrieval
                user_role=user_role,
                accessible_docs=[],
                limit=limit  # Pass custom limit
            )
            search_time = time.time() - search_start
            
            print(f"📚 Retrieved {len(search_results)} documents in {search_time:.2f}s")
            for i, doc in enumerate(search_results):
                print(f"  - Doc {i+1}: Source={doc.get('metadata', {}).get('source', 'unknown')}")
                print(f"    Content: {doc.get('content', '')[:100]}...")
            
            # STEP 3.5: Handle access limitations for financial data
            access_note = ""
            if financial_query and isinstance(user_role, Role) and user_role != Role.FINANCE and user_role != Role.C_LEVEL:
                access_note = (
                    f"\nNote: As a {role_value} role, you have limited access to detailed financial information. "
                    f"While I can provide general information about company finances, specific revenue figures "
                    f"and detailed financial reports are restricted to Finance and C-level roles. "
                    f"If you need complete financial data, please contact the Finance department or a C-level executive."
                )
                
                # If we didn't find any documents, provide some general info about the company
                if len(search_results) == 0:
                    general_financial_context = (
                        "The company maintains financial records that track revenue, expenses, and profitability "
                        "across various departments and business units. Financial reports are generated quarterly "
                        "and annually, with detailed breakdowns available to Finance department and C-level executives. "
                        "Marketing teams typically have access to budget allocations and marketing-specific financial metrics, "
                        "but not to company-wide financial details."
                    )
                    search_results.append({
                        "content": general_financial_context,
                        "metadata": {"source": "general_company_info"},
                        "similarity": 0.75
                    })
            
            # STEP 4: Translate retrieved content back to query language if needed
            print(f"\n🔄 STEP 4: Translating results (if needed)...")
            if query_lang != 'en' and confidence > 0.5:
                print(f"🔄 Translating {len(search_results)} documents to {query_lang}")
                search_results = translate_content_blocks(search_results, target_lang=query_lang)
                print(f"🔄 Translation complete")
                
                # Translate access note if exists
                if access_note:
                    access_note = translate_text(access_note, target_lang=query_lang)
            else:
                print(f"🔄 No translation needed for results")
                
            # STEP 5: Format documents for the LLM context
            print(f"\n📝 STEP 5: Formatting context...")
            context_str = self._format_context_for_llm(search_results)
            if access_note:
                context_str += access_note
            print(f"📝 Context length: {len(context_str)} chars")
            print(f"📝 Context preview: {context_str[:150]}...")
            
            # STEP 6: Generate prompt with user query and context
            print(f"\n📋 STEP 6: Building prompt...")
            prompt = self._generate_prompt(query, context_str, conversation_history, user_role)
            print(f"📋 Prompt length: {len(prompt)} chars")
            
            # STEP 7: Generate text using the LLM
            print(f"\n🤖 STEP 7: Generating response with LLM...")
            llm_start = time.time()
            generated_text = self._generate_text_with_llm(prompt)
            llm_time = time.time() - llm_start
            print(f"🤖 Response generated in {llm_time:.2f}s")
            print(f"🤖 Response: {generated_text[:150]}...")
            
            # STEP 8: Ensure language consistency in the response
            print(f"\n🌐 STEP 8: Ensuring language consistency...")
            if query_lang != 'en' and confidence > 0.5:
                # Check if the response is actually in the target language
                response_lang, resp_conf = detect_language(generated_text)
                if response_lang != query_lang and resp_conf > 0.5:
                    print(f"🌐 Response not in {query_lang}, translating from {response_lang}...")
                    generated_text = translate_text(generated_text, target_lang=query_lang, source_lang='en')
            
            # Special handling for Bangla language responses
            if query_lang == 'bn' and (generated_text.startswith("I don't have") or 
                                      "I don't have that information" in generated_text):
                generated_text = "আমার কাছে এই তথ্য নেই। আমি কি আপনাকে অন্য কিছু সাহায্য করতে পারি?"
            
            # Special handling for "how are you" in Bangla
            if query_lang == 'bn' and query.lower().strip() in ["kemon acho", "kemon achen", "কেমন আছেন", "কেমন আছো"]:
                generated_text = "আমি ভালো আছি, ধন্যবাদ। আপনি কেমন আছেন? আমি কি আপনাকে সাহায্য করতে পারি?"
            
            # Determine confidence level for response
            response_confidence = 0.9  # Default high confidence
            if len(search_results) == 0:
                response_confidence = 0.7  # Lower confidence when no relevant documents
            elif "I don't have that specific information" in generated_text:
                response_confidence = 0.7  # Lower confidence for "don't know" responses
                
            # Prepare response with metadata
            response = {
                "text": generated_text,
                "metadata": {
                    "query_language": query_lang,
                    "sources": [result.get("metadata", {}).get("source", "unknown") for result in search_results],
                    "document_count": len(search_results),
                    "retrieval_time": f"{search_time:.2f}s",
                    "llm_time": f"{llm_time:.2f}s",
                    "total_time": f"{search_time + llm_time:.2f}s",
                    "confidence": response_confidence  # Add confidence score
                }
            }
            
            # Cache the response, but only if we're using a cache key
            if cache_key:
                self._response_cache[cache_key] = response
            
            print(f"\n✅ Pipeline complete: Found {len(search_results)} docs, generated {len(generated_text)} chars")
            
            return response
        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            print(f"\n❌ ERROR in RAG pipeline: {e}")
            import traceback
            print(traceback.format_exc())
            return {"text": "I encountered an error processing your request. Please try again.", "metadata": {}}

    def _format_context_for_llm(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Format search results into a context string for the LLM.
        
        Args:
            search_results: List of document chunks from the vector store
            
        Returns:
            Formatted context string
        """
        if not search_results:
            return "No relevant information found."
            
        context_parts = []
        
        for i, result in enumerate(search_results):
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            source = metadata.get("source", "Unknown source")
            
            # Format the chunk with its metadata
            formatted_chunk = f"Document {i+1} (Source: {source}):\n{content}\n"
            context_parts.append(formatted_chunk)
            
        return "\n".join(context_parts)
        
    def _generate_prompt(self, query: str, context: str, conversation_history: List[Dict[str, str]], user_role: Optional[Role] = None) -> str:
        """
        Generate a prompt for the LLM using the query, context, and conversation history.
        
        Args:
            query: User query
            context: Formatted context from relevant documents
            conversation_history: Previous conversation turns
            user_role: The role of the user to personalize responses
            
        Returns:
            Complete prompt for the LLM
        """
        # Format conversation history more efficiently
        formatted_history = ""
        
        if conversation_history:
            # Focus only on recent and relevant conversation turns
            relevant_turns = []
            
            # If we have a lot of history, be selective
            if len(conversation_history) > 3:
                # Always include the most recent turn
                if len(conversation_history) > 0:
                    relevant_turns.append(conversation_history[-1])
                
                # Check for query-related terms in history
                query_terms = set(query.lower().split())
                for turn in conversation_history[-5:-1]:  # Skip the most recent one we already added
                    user_msg = turn.get("user", "").lower()
                    # Include turns that share terms with current query
                    if any(term in user_msg for term in query_terms if len(term) > 3):
                        relevant_turns.append(turn)
            else:
                # For short conversations, include everything
                relevant_turns = conversation_history
            
            # Format the selected turns
            for turn in relevant_turns[-3:]:  # Limit to last 3 relevant turns
                user_msg = turn.get("user", "")
                assistant_msg = turn.get("assistant", "")
                if user_msg:
                    formatted_history += f"User: {user_msg}\n"
                if assistant_msg:
                    # Truncate very long assistant responses in history
                    if len(assistant_msg) > 300:
                        assistant_msg = assistant_msg[:250] + "... [truncated]"
                    formatted_history += f"Assistant: {assistant_msg}\n"
        
        # Get role-specific system prompt
        role_prompts = {
            Role.EMPLOYEE: "You are a helpful general assistant for employees.",
            Role.FINANCE: "You are a financial assistant with expertise in financial data and analysis.",
            Role.HR: "You are an HR assistant with expertise in employee policies and information.",
            Role.ENGINEERING: "You are a technical assistant with expertise in engineering documentation.",
            Role.MARKETING: "You are a marketing assistant with expertise in marketing campaigns and analytics.",
            Role.C_LEVEL: "You are an executive assistant with comprehensive company access."
        }
        
        role_description = ""
        if user_role:
            if isinstance(user_role, Role):
                role_description = role_prompts.get(user_role, role_prompts[Role.EMPLOYEE])
            elif isinstance(user_role, str):
                role_key = getattr(Role, user_role.upper(), None)
                if role_key:
                    role_description = role_prompts.get(role_key, role_prompts[Role.EMPLOYEE])
        
        if not role_description:
            role_description = "You are FinSolve AI, an intelligent assistant for business operations."
        
        # Build the complete prompt with FinSolve AI identity and improved instructions
        system_prompt = f"""You are FinSolve AI, an intelligent assistant for business operations. {role_description}

IMPORTANT INSTRUCTIONS:
1. Always prioritize information from the provided context when answering questions.
2. Be professional, direct, and concise in your responses.
3. If the answer is clearly found in the context, use that information confidently.
4. When information is not in the context, clearly state "I don't have that specific information" rather than making up answers.
5. Never fabricate financial data, company policies, or employee information.
6. If asked about a previous conversation and you don't have enough context, politely ask for clarification.
7. Focus on answering only what was asked, without adding unnecessary information.
8. For numerical data, use precise figures from the context rather than generalizations.

You have access to company documents including financial reports, employee information, HR policies, and departmental data."""
        
        prompt = f"{system_prompt}\n\n"
        
        # Add context - always include context before history for better focus
        if context and context.strip() != "No relevant information found.":
            prompt += f"CONTEXT INFORMATION:\n{context}\n\n"
        else:
            prompt += "CONTEXT: No specific company information was found for this query.\n\n"
        
        # Add conversation history if available
        if formatted_history:
            prompt += f"CONVERSATION HISTORY:\n{formatted_history}\n"
        
        # Add the current query with clear separation
        prompt += f"CURRENT QUERY: {query}\n"
        prompt += "FINSOLVE AI RESPONSE:"
        
        return prompt
        
    def _generate_text_with_llm(self, prompt: str) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: The complete prompt for the LLM
            
        Returns:
            Generated text
        """
        try:
            # Use the existing Ollama integration that's already working
            return self._call_ollama_gpu_optimized(prompt)
        except Exception as e:
            logger.error(f"Error generating text with LLM: {e}")
            return "I encountered an error generating a response. Please try again."