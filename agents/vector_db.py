"""Vector Database operations using ChromaDB."""

import chromadb
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from utils.config import Config
from utils.logging import logger

class VectorDB:
    """ChromaDB wrapper for managing embeddings and retrieval."""
    
    def __init__(self, collection_name: str = "support-knowledge"):
        """Initialize ChromaDB and embedding model."""
        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # BM25 for keyword search (will be built when indexing)
        self.bm25_texts: List[str] = []
        self.bm25_model: Optional[BM25Okapi] = None
        self.bm25_ids: List[str] = []
        
        logger.info(f"Initialized VectorDB with embedding model: {Config.EMBEDDING_MODEL}")
    
    def add_documents(self, 
                     documents: List[Dict[str, Any]],
                     batch_size: int = 100) -> None:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of dicts with 'id', 'text', and 'metadata'
            batch_size: Number of documents to process at once
        """
        logger.info(f"Adding {len(documents)} documents to ChromaDB...")
        
        # Extract texts and prepare BM25
        self.bm25_texts = []
        self.bm25_ids = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            # Prepare batch data
            ids = []
            embeddings = []
            metadatas = []
            documents_text = []
            
            for doc in batch:
                ids.append(doc["id"])
                documents_text.append(doc["text"])
                embeddings.append(self.embedding_model.encode(doc["text"]))
                metadatas.append(doc.get("metadata", {}))
                
                # Store for BM25
                self.bm25_texts.append(doc["text"])
                self.bm25_ids.append(doc["id"])
            
            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings,
                metadatas=metadatas,
                documents=documents_text
            )
            
            logger.debug(f"Batch {i//batch_size + 1}: Added {len(batch)} documents")
        
        # Build BM25 index
        if self.bm25_texts:
            tokenized = [text.lower().split() for text in self.bm25_texts]
            self.bm25_model = BM25Okapi(tokenized)
            logger.info(f"Built BM25 index with {len(self.bm25_texts)} documents")
    
    def retrieve(self, 
                query: str,
                top_k: int = 5,
                metadata_filter: Optional[Dict[str, Any]] = None,
                hybrid: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve documents using semantic and/or keyword search.
        
        Args:
            query: Query string
            top_k: Number of results to return
            metadata_filter: Optional metadata filtering (e.g., {"category": "billing"})
            hybrid: If True, combine semantic and BM25 scores
        
        Returns:
            List of documents with scores
        """
        query_embedding = self.embedding_model.encode(query)
        
        # Semantic search
        semantic_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k * 2,  # Get more to allow for filtering
            where=metadata_filter if metadata_filter else None,
            include=["documents", "metadatas", "distances", "embeddings"]
        )
        
        results = []
        
        if semantic_results["ids"] and len(semantic_results["ids"]) > 0:
            for i, doc_id in enumerate(semantic_results["ids"][0]):
                # ChromaDB returns distances, convert to similarity (1 - distance for cosine)
                similarity = 1 - semantic_results["distances"][0][i]
                
                results.append({
                    "id": doc_id,
                    "text": semantic_results["documents"][0][i],
                    "metadata": semantic_results["metadatas"][0][i],
                    "semantic_score": similarity,
                    "bm25_score": 0.0,
                    "hybrid_score": similarity
                })
        
        # BM25 search if hybrid mode
        if hybrid and self.bm25_model:
            query_tokens = query.lower().split()
            bm25_scores = self.bm25_model.get_scores(query_tokens)
            
            # Normalize BM25 scores to 0-1 range
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
            bm25_scores = bm25_scores / max_bm25
            
            # Create dict for quick lookup
            bm25_dict = {doc_id: score for doc_id, score in zip(self.bm25_ids, bm25_scores)}
            
            # Merge BM25 scores with semantic results
            for result in results:
                bm25_score = bm25_dict.get(result["id"], 0.0)
                result["bm25_score"] = bm25_score
                # Hybrid: average of semantic and BM25
                result["hybrid_score"] = (result["semantic_score"] + bm25_score) / 2
            
            # Sort by hybrid score
            results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        else:
            # Sort by semantic score
            results.sort(key=lambda x: x["semantic_score"], reverse=True)
        
        logger.debug(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        return results[:top_k]
    
    def rerank(self, 
              query: str,
              documents: List[Dict[str, Any]],
              top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank documents using a cross-encoder model.
        
        Args:
            query: Query string
            documents: List of documents to rerank
            top_k: Number of top results to return
        
        Returns:
            Reranked documents with new scores
        """
        try:
            from sentence_transformers import CrossEncoder
            
            reranker = CrossEncoder(Config.RERANK_MODEL)
            
            # Prepare pairs for cross-encoder
            pairs = [[query, doc["text"]] for doc in documents]
            
            # Get scores
            scores = reranker.predict(pairs)
            
            # Normalize scores to 0-1
            min_score = scores.min()
            max_score = scores.max()
            normalized_scores = (scores - min_score) / (max_score - min_score + 1e-8)
            
            # Update documents with reranking scores
            for doc, score in zip(documents, normalized_scores):
                doc["rerank_score"] = float(score)
            
            # Sort by rerank score
            documents.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            logger.debug(f"Reranked {len(documents)} documents")
            return documents[:top_k]
            
        except ImportError:
            logger.warning("CrossEncoder not available, using original scores")
            return documents[:top_k]
    
    def hybrid_retrieve(self,
                       query: str,
                       top_k: int = 5,
                       metadata_filter: Optional[Dict[str, Any]] = None,
                       rerank: bool = True) -> List[Dict[str, Any]]:
        """
        Full hybrid retrieval pipeline: semantic + BM25 + reranking.
        
        This is the recommended retrieval method for production RAG.
        """
        # Step 1: Retrieve with hybrid search
        results = self.retrieve(query, top_k=top_k * 2, metadata_filter=metadata_filter, hybrid=True)
        
        # Step 2: Rerank (optional but recommended)
        if rerank and results:
            results = self.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]
        
        return results
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        # Delete collection and recreate
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        self.bm25_texts = []
        self.bm25_model = None
        self.bm25_ids = []
        logger.info("Cleared VectorDB")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        count = self.collection.count()
        return {
            "collection_name": self.collection.name,
            "document_count": count,
            "embedding_dim": self.embedding_dim,
            "embedding_model": Config.EMBEDDING_MODEL
        }
