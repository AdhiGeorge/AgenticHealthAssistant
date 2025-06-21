"""Qdrant vector store integration for medical context search."""

import hashlib
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue
)
from sentence_transformers import SentenceTransformer
from medical_analysis.utils.logger import get_logger
from medical_analysis.utils.config import get_config

class MedicalVectorStore:
    """Qdrant-based vector store for medical context and conversation search."""
    
    def __init__(self, collection_name: str = "medical_context"):
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.collection_name = collection_name
        
        # Initialize Qdrant client
        try:
            self.client = QdrantClient("localhost", port=6333)
            self.logger.info("Connected to Qdrant vector database")
        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {e}")
            self.client = None
        
        # Initialize sentence transformer for embeddings
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Loaded sentence transformer model")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
        
        # Initialize collection if needed
        if self.client:
            self._init_collection()
    
    def _init_collection(self):
        """Initialize the Qdrant collection for medical context."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # all-MiniLM-L6-v2 embedding size
                        distance=Distance.COSINE
                    )
                )
                self.logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                self.logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Qdrant collection: {e}")
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text."""
        if not self.embedding_model or not text:
            return None
        
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def _generate_id(self, content: str, content_type: str, session_id: str) -> str:
        """Generate unique ID for vector point."""
        hash_input = f"{session_id}_{content_type}_{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def store_medical_context(self, session_id: str, content: str, content_type: str, 
                            metadata: Optional[Dict] = None) -> bool:
        """Store medical context in vector database."""
        if not self.client or not self.embedding_model:
            self.logger.warning("Vector store not available")
            return False
        
        try:
            # Generate embedding
            embedding = self._generate_embedding(content)
            if not embedding:
                return False
            
            # Generate unique ID
            point_id = self._generate_id(content, content_type, session_id)
            
            # Prepare metadata
            point_metadata = {
                "session_id": session_id,
                "content_type": content_type,
                "content_length": len(content),
                "timestamp": datetime.now().isoformat(),
                "content_preview": content[:200] + "..." if len(content) > 200 else content
            }
            
            if metadata:
                point_metadata.update(metadata)
            
            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=point_metadata
            )
            
            # Store in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            self.logger.info(f"Stored {content_type} context for session {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store medical context: {e}")
            return False
    
    def search_similar_context(self, query: str, session_id: Optional[str] = None, 
                             limit: int = 5, threshold: float = 0.7) -> List[Dict]:
        """Search for similar medical context."""
        if not self.client or not self.embedding_model:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                return []
            
            # Prepare search filter
            search_filter = None
            if session_id:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=session_id)
                        )
                    ]
                )
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit,
                score_threshold=threshold
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "content_type": result.payload.get("content_type"),
                    "content_preview": result.payload.get("content_preview"),
                    "session_id": result.payload.get("session_id"),
                    "timestamp": result.payload.get("timestamp"),
                    "metadata": {k: v for k, v in result.payload.items() 
                               if k not in ["content_type", "content_preview", "session_id", "timestamp"]}
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search similar context: {e}")
            return []
    
    def get_session_context(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get all context for a specific session."""
        if not self.client:
            return []
        
        try:
            # Search for all points in the session
            search_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=session_id)
                        )
                    ]
                ),
                limit=limit
            )
            
            results = []
            for point in search_results[0]:  # scroll returns (points, next_page_offset)
                results.append({
                    "id": point.id,
                    "content_type": point.payload.get("content_type"),
                    "content_preview": point.payload.get("content_preview"),
                    "timestamp": point.payload.get("timestamp"),
                    "metadata": {k: v for k, v in point.payload.items() 
                               if k not in ["content_type", "content_preview", "session_id", "timestamp"]}
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get session context: {e}")
            return []
    
    def delete_session_context(self, session_id: str) -> bool:
        """Delete all context for a specific session."""
        if not self.client:
            return False
        
        try:
            # Find all points for the session
            search_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=session_id)
                        )
                    ]
                ),
                limit=1000  # Large limit to get all points
            )
            
            if search_results[0]:
                point_ids = [point.id for point in search_results[0]]
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=point_ids
                )
                self.logger.info(f"Deleted {len(point_ids)} context points for session {session_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete session context: {e}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        if not self.client:
            return {}
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": collection_info.name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": str(collection_info.config.params.vectors.distance)
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {} 