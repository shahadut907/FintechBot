"""
ENHANCED Vector Store with Improved GPU Detection and CPU Optimization
"""
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import uuid
import json
import hashlib
import time
import threading
from functools import lru_cache
import torch
import gc
from src.config.settings import settings
from src.config.roles import Role, Department, ROLE_PERMISSIONS
from src.backend.utils.document_processor import DocumentChunk
import re
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a logger for this module
logger = logging.getLogger(__name__)

class VectorStore:
    """Enhanced AI Memory System with Better GPU/CPU Handling"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, collection_name: str = "company_docs", persist_directory: str = None):
        """
        Initialize the vector store with specified collection and persistence settings
        
        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory for persistence
        """
        # Set up default persist directory if none provided
        if persist_directory is None:
            persist_directory = os.path.join(os.path.dirname(__file__), "..", "..", "data", "embeddings", "chroma_db")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            
            # Create collection if it doesn't exist
            try:
                self.collection = self.client.get_collection(name=collection_name)
                logger.info(f"Connected to existing collection: {collection_name}")
            except ValueError:
                # Create a new collection with the sentence transformer embedding function
                self.collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self._get_embedding_function()
                )
                logger.info(f"Created new collection: {collection_name}")
                
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
        
        if self._initialized:
            return
            
        print("🧠 Initializing Enhanced AI Memory System...")
        
        # Performance tracking
        self._embedding_cache = {}
        self._max_cache_size = 1000
        self._cache_stats = {"hits": 0, "misses": 0}
        self._performance_stats = {
            "total_embeddings": 0,
            "avg_embedding_time": 0,
            "cache_hit_rate": 0
        }
        
        # Setup embedding model with enhanced GPU/CPU detection
        self._setup_embedding_model()
        
        self._initialized = True
        print("✅ Enhanced AI Memory System ready")
    
    def _setup_embedding_model(self):
        """Enhanced embedding model setup with better GPU/CPU handling"""
        print(f"📥 Loading Enhanced Embeddings Model: {settings.local_embeddings_model}")
        
        # Detailed device detection
        self._detect_compute_device()
        
        # Load model with device-specific optimizations
        try:
            self.embeddings_model = SentenceTransformer(
                settings.local_embeddings_model,
                device=self.device
            )
            
            # Apply device-specific optimizations
            self._optimize_model_for_device()
            
            # Warm up the model
            self._warmup_embedding_model()
            
            print(f"✅ Embeddings model loaded on {self.device}")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise e
    
    def _detect_compute_device(self):
        """Enhanced device detection with detailed diagnostics"""
        print("🔍 Detecting optimal compute device...")
        
        try:
            if torch.cuda.is_available():
                self.device = "cuda"
                self.cuda_available = True
                
                # Detailed GPU info
                gpu_name = torch.cuda.get_device_name()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                gpu_count = torch.cuda.device_count()
                
                print(f"✅ CUDA GPU detected: {gpu_name}")
                print(f"📊 GPU Memory: {gpu_memory:.1f} GB")
                print(f"🔢 GPU Count: {gpu_count}")
                print(f"🔧 CUDA Version: {torch.version.cuda}")
                
                # Check available memory
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"💾 GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
                
            else:
                self.device = "cpu"
                self.cuda_available = False
                
                print("⚠️ CUDA not available - using optimized CPU")
                print("🔧 CPU Info:")
                print(f"   Threads available: {torch.get_num_threads()}")
                print(f"   MKL available: {torch.backends.mkl.is_available()}")
                print(f"   OpenMP available: {torch.backends.openmp.is_available()}")
                
                # CPU optimizations
                self._apply_cpu_optimizations()
                
        except Exception as e:
            print(f"❌ Device detection error: {e}")
            self.device = "cpu"
            self.cuda_available = False
    
    def _apply_cpu_optimizations(self):
        """Apply aggressive CPU optimizations when GPU is not available"""
        import os
        
        # Set environment variables for optimal CPU performance
        optimizations = {
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8", 
            "OPENBLAS_NUM_THREADS": "8",
            "VECLIB_MAXIMUM_THREADS": "8",
            "NUMEXPR_NUM_THREADS": "8"
        }
        
        for var, value in optimizations.items():
            os.environ[var] = value
            
        # PyTorch specific optimizations
        torch.set_num_threads(8)
        torch.set_num_interop_threads(4)
        
        print("🚀 Applied aggressive CPU optimizations")
    
    def _optimize_model_for_device(self):
        """Apply device-specific model optimizations"""
        if self.cuda_available:
            print("🚀 Applying GPU optimizations...")
            
            # Set model to eval mode
            self.embeddings_model.eval()
            
            # Try half precision if supported
            try:
                self.embeddings_model.half()
                self._use_half_precision = True
                print("✅ Enabled half-precision (FP16) inference")
            except Exception as e:
                self._use_half_precision = False
                print(f"⚠️ Half-precision not supported: {e}")
            
            # CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
        else:
            print("🔧 Applying CPU optimizations...")
            self._use_half_precision = False
            
            # CPU-specific optimizations
            self.embeddings_model.eval()
    
    def _warmup_embedding_model(self):
        """Comprehensive model warmup"""
        print("🔥 Warming up embedding model...")
        
        warmup_texts = [
            "financial report analysis",
            "engineering documentation query", 
            "hr policy information",
            "marketing campaign data",
            "executive summary report"
        ]
        
        start_time = time.perf_counter()
        
        try:
            # Single embedding warmup
            for text in warmup_texts:
                _ = self.embeddings_model.encode(
                    text,
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    batch_size=1,
                    device=self.device
                )
            
            # Batch embedding warmup
            _ = self.embeddings_model.encode(
                warmup_texts,
                convert_to_tensor=False,
                show_progress_bar=False,
                batch_size=len(warmup_texts),
                device=self.device
            )
            
            warmup_time = time.perf_counter() - start_time
            print(f"✅ Model warmed up in {warmup_time:.2f}s")
            
        except Exception as e:
            print(f"⚠️ Warmup failed: {e}")
    
    def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Optimized batch embedding generation"""
        if not texts:
            return []
        
        batch_start = time.perf_counter()
        
        try:
            # Determine optimal batch size based on device
            if self.cuda_available:
                batch_size = min(64, len(texts))  # Larger batches for GPU
            else:
                batch_size = min(16, len(texts))  # Smaller batches for CPU
            
            # Generate embeddings with optimizations
            embeddings = self.embeddings_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_tensor=False,
                normalize_embeddings=True,
                device=self.device
            )
            
            batch_time = time.perf_counter() - batch_start
            
            # Update performance stats
            self._performance_stats["total_embeddings"] += len(texts)
            avg_time = batch_time / len(texts)
            
            if self._performance_stats["avg_embedding_time"] == 0:
                self._performance_stats["avg_embedding_time"] = avg_time
            else:
                # Exponential moving average
                self._performance_stats["avg_embedding_time"] = (
                    0.9 * self._performance_stats["avg_embedding_time"] + 0.1 * avg_time
                )
            
            if settings.debug:
                print(f"🚀 Batch embedding: {len(texts)} texts in {batch_time:.3f}s ({avg_time*1000:.1f}ms/text)")
            
            return embeddings.tolist()
            
        except Exception as e:
            print(f"❌ Batch embedding error: {e}")
            # Fallback to individual embeddings
            return [self._get_single_embedding(text) for text in texts]
    
    @lru_cache(maxsize=2000)
    def _get_cached_embedding(self, query: str) -> List[float]:
        """Enhanced embedding caching with performance tracking"""
        # Check manual cache first
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        if query_hash in self._embedding_cache:
            self._cache_stats["hits"] += 1
            return self._embedding_cache[query_hash]
        
        self._cache_stats["misses"] += 1
        
        # Generate new embedding
        embedding = self._get_single_embedding(query)
        
        # Manage cache size
        if len(self._embedding_cache) >= self._max_cache_size:
            # Remove oldest 20% of entries
            remove_count = self._max_cache_size // 5
            oldest_keys = list(self._embedding_cache.keys())[:remove_count]
            for key in oldest_keys:
                del self._embedding_cache[key]
        
        self._embedding_cache[query_hash] = embedding
        return embedding
    
    def _get_single_embedding(self, text: str) -> List[float]:
        """Optimized single embedding generation"""
        embed_start = time.perf_counter()
        
        try:
            embedding = self.embeddings_model.encode(
                text,
                batch_size=1,
                show_progress_bar=False,
                convert_to_tensor=False,
                normalize_embeddings=True,
                device=self.device
            )
            
            embed_time = time.perf_counter() - embed_start
            
            if settings.debug and embed_time > 0.5:
                print(f"⚠️ Slow embedding: {embed_time:.3f}s for text length {len(text)}")
            
            return embedding.tolist()
            
        except Exception as e:
            print(f"❌ Single embedding error: {e}")
            # Return zero vector as fallback
            return [0.0] * 384

    def search_documents(self, query: str, user_role: Role = Role.EMPLOYEE, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Search documents based on query with RBAC filtering
        
        Args:
            query: The search query
            user_role: User role for RBAC
            limit: Maximum number of results to return
            
        Returns:
            List of document dictionaries with content and metadata
        """
        try:
            # Get departments accessible to the user role
            accessible_departments = self._get_accessible_departments(user_role)
            
            # Generate embedding for query
            query_embedding = self._get_cached_embedding(query)
            if query_embedding is None:
                return []
            
            # Search with department filtering
            search_kwargs = {
                "k": min(limit * 3, 20),  # Get more than needed for filtering
                "where": {"department": {"$in": accessible_departments}},
                "include": ["metadatas", "documents", "distances"]
            }
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                **search_kwargs
            )
            
            # Process results
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            # Build result list
            document_list = []
            for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
                # Convert distance to similarity
                similarity = 1.0 - min(dist, 1.0)
                
                # Skip very poor matches
                if similarity < 0.3:
                    continue
                
                document_list.append({
                    "content": doc,
                    "metadata": meta,
                    "similarity": similarity,
                    "source": meta.get("source", "unknown")
                })
            
            # Sort by similarity
            document_list.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Return top results
            return document_list[:limit]
            
        except Exception as e:
            logger.error(f"Error in search_documents: {e}")
            return []

    def _get_accessible_departments(self, user_role: Role) -> List[str]:
        """Get accessible departments for role"""
        user_permissions = ROLE_PERMISSIONS.get(user_role)
        if not user_permissions:
            return ["general"]
        
        return [dept.value for dept in user_permissions.allowed_departments]

    def _get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self._cache_stats["hits"] + self._cache_stats["misses"]
        return self._cache_stats["hits"] / max(total, 1)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Enhanced performance statistics"""
        return {
            "device": self.device,
            "cuda_available": self.cuda_available,
            "half_precision": getattr(self, '_use_half_precision', False),
            "cache_hit_rate": f"{self._get_cache_hit_rate():.2%}",
            "cache_size": len(self._embedding_cache),
            "cache_hits": self._cache_stats["hits"],
            "cache_misses": self._cache_stats["misses"],
            "total_embeddings": self._performance_stats["total_embeddings"],
            "avg_embedding_time": f"{self._performance_stats['avg_embedding_time']*1000:.1f}ms",
            "model_name": settings.local_embeddings_model
        }

    # ... (rest of the methods remain the same: add_documents, similarity_search, etc.)
    def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks with enhanced performance tracking"""
        print(f"📚 Adding {len(chunks)} document chunks with enhanced processing...")
        
        add_start = time.perf_counter()
        
        try:
            batch_size = 100 if self.cuda_available else 50
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                batch_start = time.perf_counter()
                
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(chunks))
                batch_chunks = chunks[start_idx:end_idx]
                
                # Prepare data
                documents = []
                embeddings = []
                metadatas = []
                ids = []
                
                texts = [chunk.content for chunk in batch_chunks]
                
                # Generate embeddings in batch
                batch_embeddings = self._generate_batch_embeddings(texts)
                
                # Process chunks
                for i, chunk in enumerate(batch_chunks):
                    chunk_id = f"{chunk.metadata['source']}_{chunk.metadata['chunk_id']}_{uuid.uuid4().hex[:8]}"
                    
                    dept = chunk.metadata["department"]
                    dept_value = dept.value if hasattr(dept, 'value') else str(dept)
                    
                    metadata = {
                        "source": chunk.metadata["source"],
                        "department": dept_value,
                        "department_enum": dept_value,
                        "doc_type": chunk.metadata.get("doc_type", "unknown"),
                        "content_type": chunk.metadata.get("content_type", "text"),
                        "chunk_id": str(chunk.metadata["chunk_id"])
                    }
                    
                    documents.append(chunk.content)
                    embeddings.append(batch_embeddings[i])
                    metadatas.append(metadata)
                    ids.append(chunk_id)
                
                # Add to ChromaDB
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                
                batch_time = time.perf_counter() - batch_start
                print(f"   Batch {batch_idx + 1}/{total_batches}: {len(batch_chunks)} chunks in {batch_time:.2f}s")
            
            total_time = time.perf_counter() - add_start
            print(f"✅ Enhanced processing: {len(chunks)} chunks in {total_time:.2f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Enhanced document adding error: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Enhanced collection statistics"""
        try:
            collection_count = self.collection.count()
            
            # Get sample for analysis
            sample_size = min(100, collection_count)
            sample_results = self.collection.get(limit=sample_size, include=["metadatas"])
            
            departments = {}
            doc_types = {}
            sources = {}
            
            for metadata in sample_results["metadatas"]:
                dept = metadata.get("department", "unknown")
                doc_type = metadata.get("doc_type", "unknown")
                source = metadata.get("source", "unknown")
                
                departments[dept] = departments.get(dept, 0) + 1
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                sources[source] = sources.get(source, 0) + 1
            
            return {
                "total_documents": collection_count,
                "departments": departments,
                "document_types": doc_types,
                "sources": sources,
                "performance": self.get_performance_stats(),
                "sample_size": sample_size
            }
            
        except Exception as e:
            return {"error": str(e), "total_documents": 0}

    def similarity_search(self, query: str, k: int = 5) -> List[Any]:
        """Standard similarity search with enhanced performance"""
        try:
            query_embedding = self._get_cached_embedding(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            docs = []
            if results["documents"] and len(results["documents"]) > 0:
                for i in range(len(results["documents"][0])):
                    class MockDocument:
                        def __init__(self, content, metadata):
                            self.page_content = content
                            self.metadata = metadata
                    
                    doc = MockDocument(
                        content=results["documents"][0][i],
                        metadata=results["metadatas"][0][i]
                    )
                    docs.append(doc)
            
            return docs
            
        except Exception as e:
            print(f"❌ Similarity search error: {e}")
            return []

    def clear_collection(self):
        """Clear collection with cache cleanup"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self._get_embedding_function()
            )
            
            # Clear all caches
            self._embedding_cache.clear()
            self._cache_stats = {"hits": 0, "misses": 0}
            self._performance_stats = {
                "total_embeddings": 0,
                "avg_embedding_time": 0,
                "cache_hit_rate": 0
            }
            
            print("🗑️ Collection and caches cleared successfully")
            
        except Exception as e:
            print(f"❌ Error clearing collection: {e}")

    def get_unique_sources(self) -> List[str]:
        """Return a list of unique source filenames currently stored in the collection."""
        try:
            total_docs = self.collection.count()
            if total_docs == 0:
                return []

            # Fetch only metadata to reduce memory overhead
            results = self.collection.get(limit=total_docs, include=["metadatas"])
            sources = set()
            for meta in results["metadatas"]:
                source = meta.get("source")
                if source:
                    sources.add(source)
            return list(sources)
        except Exception as e:
            print(f"⚠️ Unable to fetch unique sources: {e}")
            return []

    def delete_documents_by_sources(self, sources: List[str]) -> int:
        """Delete all document chunks whose metadata.source is in the provided list.
        Returns number of deleted chunks (best-effort)."""
        if not sources:
            return 0
        try:
            self.collection.delete(where={"source": {"$in": sources}})
            # Best-effort count removal stats – may require another count
            print(f"🗑️ Removed vectors for {len(sources)} stale sources: {sources}")

            # Clean internal caches because underlying data changed
            self._embedding_cache.clear()
            self._cache_stats = {"hits": 0, "misses": 0}
            return len(sources)
        except Exception as e:
            print(f"❌ Error deleting documents for sources {sources}: {e}")
            return 0

    def keyword_search(self, query: str, user_role: Role, limit: int = 5) -> List[Dict[str, Any]]:
        """Enhanced keyword search as fallback when semantic search fails"""
        print(f"\n🔍 DEBUG: Running enhanced keyword search for: '{query}'")
        
        try:
            # Get departments accessible to the user role
            accessible_departments = self._get_accessible_departments(user_role)
            
            # Extract important terms for search
            search_terms = self._extract_search_terms(query)
            if not search_terms:
                print(f"🔍 No valid search terms extracted from query")
                return []
            
            print(f"🔍 Using search terms: {search_terms}")
            
            # Get all documents and filter manually for better control
            all_docs = self.collection.get(
                where={"department": {"$in": accessible_departments}},
                include=["metadatas", "documents"]
            )
            
            documents = all_docs.get("documents", [[]])[0]
            metadatas = all_docs.get("metadatas", [[]])[0]
            
            if not documents:
                return []
            
            # Score documents based on term frequency
            scored_docs = []
            for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                if not doc or not isinstance(doc, str):
                    continue
                
                # Calculate document score based on term matches
                score = 0
                doc_lower = doc.lower()
                
                # Score exact matches more highly
                for term in search_terms:
                    if term.lower() in doc_lower:
                        # Count occurrences (with diminishing returns)
                        count = doc_lower.count(term.lower())
                        score += min(count * 0.2, 1.0)
                        
                        # Boost score for exact matches
                        if re.search(r'\b' + re.escape(term.lower()) + r'\b', doc_lower):
                            score += 0.5
                
                # Only include documents with at least one matching term
                if score > 0:
                    scored_docs.append({
                        "content": doc,
                        "metadata": meta,
                        "similarity": score,
                        "source": meta.get("source", "unknown")
                    })
            
            # Sort by score and limit results
            scored_docs.sort(key=lambda x: x["similarity"], reverse=True)
            results = scored_docs[:limit]
            
            print(f"🔍 Keyword search found {len(results)} relevant documents")
            return results
            
        except Exception as e:
            print(f"🔍 ERROR in keyword search: {e}")
            return []
            
    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract meaningful search terms from a query"""
        # Remove common stopwords for better keyword matching
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with',
                   'by', 'about', 'as', 'into', 'like', 'through', 'after', 'over', 'between',
                   'out', 'of', 'from', 'up', 'about', 'is', 'are', 'was', 'were', 'be', 'been',
                   'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
                   'should', 'may', 'might', 'must', 'can', 'could'}
        
        # Split query and filter out stopwords
        words = query.split()
        filtered_words = [word.strip('.,?!()[]{}":;') for word in words 
                         if word.lower() not in stopwords and len(word) > 2]
        
        # Extract named entities or special terms
        special_terms = []
        
        # Look for terms in quotes
        quoted_terms = re.findall(r'"([^"]+)"', query)
        if quoted_terms:
            special_terms.extend(quoted_terms)
        
        # Look for potential named entities (capitalized terms)
        for i, word in enumerate(words):
            if i > 0 and word[0].isupper() and len(word) > 1:
                special_terms.append(word.strip('.,?!()[]{}":;'))
        
        # Look for numbers and dates
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query)
        special_terms.extend(numbers)
        
        # Add any special terms that weren't already included
        for term in special_terms:
            if term.lower() not in [w.lower() for w in filtered_words]:
                filtered_words.append(term)
        
        return filtered_words

    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess the query to improve search quality
        
        Args:
            query: The original query
            
        Returns:
            Processed query
        """
        # Simple preprocessing - trim whitespace, lowercase
        processed = query.strip().lower()
        
        # Remove excessive punctuation but preserve important characters
        processed = re.sub(r'[^\w\s\?\.\,\-]', ' ', processed)
        
        # Remove extra spaces
        processed = re.sub(r'\s+', ' ', processed)
        
        return processed
        
    def get_accessible_roles_for_user(self, user_role: Role) -> List[str]:
        """
        Get the roles that this user can access based on their own role
        
        Args:
            user_role: The user's role
            
        Returns:
            List of role names accessible to this user
        """
        # Role hierarchy - higher roles can access lower roles' content
        role_hierarchy = {
            Role.C_LEVEL: ["C_LEVEL", "FINANCE", "HR", "ENGINEERING", "MARKETING", "EMPLOYEE"],
            Role.FINANCE: ["FINANCE", "EMPLOYEE"],
            Role.HR: ["HR", "EMPLOYEE"],
            Role.ENGINEERING: ["ENGINEERING", "EMPLOYEE"],
            Role.MARKETING: ["MARKETING", "EMPLOYEE"],
            Role.EMPLOYEE: ["EMPLOYEE"]
        }
        
        # Handle string role values
        if isinstance(user_role, str):
            try:
                user_role = Role[user_role.upper()]
            except (KeyError, AttributeError):
                return ["EMPLOYEE"]  # Default to employee access
        
        # Return accessible roles based on user's role
        return role_hierarchy.get(user_role, ["EMPLOYEE"])

    def _get_embedding_function(self):
        """
        Get the sentence transformer embedding function for Chroma
        
        Returns:
            Embedding function to use with ChromaDB
        """
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Error creating embedding function: {e}")
            return None