"""RAG Pipeline: Chunking, embedding, and retrieval."""

from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.config import Config
from utils.logging import logger

class RAGPipeline:
    """Pipeline for ingesting and chunking documents for RAG."""
    
    def __init__(self):
        """Initialize the chunking strategy."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=int(Config.CHUNK_SIZE * Config.CHUNK_OVERLAP / 100),
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        logger.info(
            f"Initialized RAG pipeline with chunk_size={Config.CHUNK_SIZE}, "
            f"overlap={Config.CHUNK_OVERLAP}%"
        )
    
    def chunk_document(self, 
                      text: str, 
                      metadata: Dict[str, Any] = None,
                      doc_index: int = 0) -> List[Dict[str, Any]]:
        """
        Chunk a document into semantic pieces.
        
        Args:
            text: Document text to chunk
            metadata: Metadata to attach to all chunks (source, category, etc.)
            doc_index: Index of this document in the batch (for unique ID generation)
        
        Returns:
            List of chunks with metadata
        """
        if not text or not text.strip():
            logger.warning("Empty document provided to chunk_document")
            return []
        
        chunks_text = self.text_splitter.split_text(text)
        chunks = []
        
        # Normalize source path: replace backslashes and spaces for safe IDs
        source = metadata.get('source', 'unknown') if metadata else 'unknown'
        source_key = source.replace('\\', '/').replace(' ', '_')
        
        for i, chunk_text in enumerate(chunks_text):
            # Include doc_index so chunks from multiple docs of same source are unique
            chunk_id = f"{source_key}__doc{doc_index}__chunk{i}"
            
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "chunk_id": i,
                "chunk_count": len(chunks_text),
                "doc_index": doc_index,
            })
            
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": chunk_metadata
            })
        
        logger.debug(f"Chunked document into {len(chunks)} pieces (source: {source_key}, doc_index: {doc_index})")
        return chunks
    
    def process_documents(self,
                         documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple documents into chunks.
        
        Args:
            documents: List of dicts with 'text' and optional 'metadata'
        
        Returns:
            List of all chunks with metadata
        """
        all_chunks = []
        
        for doc_index, doc in enumerate(documents):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            # Pass doc_index so every chunk across the batch gets a globally unique ID
            chunks = self.chunk_document(text, metadata, doc_index=doc_index)
            all_chunks.extend(chunks)
        
        logger.info(f"Processed {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text
        
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = " ".join(text.split())
        # Remove URLs
        import re
        text = re.sub(r'http\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        return text.strip()

class DocumentLoader:
    """Load documents from various sources."""
    
    @staticmethod
    def load_markdown(filepath: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Load markdown file as documents."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by headers for better chunking
            sections = content.split('\n# ')
            documents = []
            
            for section in sections:
                if section.strip():
                    doc_metadata = metadata.copy() if metadata else {}
                    doc_metadata["source"] = filepath
                    doc_metadata["type"] = "markdown"
                    
                    documents.append({
                        "text": section.strip(),
                        "metadata": doc_metadata
                    })
            
            logger.info(f"Loaded {len(documents)} sections from {filepath}")
            return documents
        
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            return []
    
    @staticmethod
    def load_json(filepath: str, text_field: str = "text", metadata_fields: List[str] = None) -> List[Dict[str, Any]]:
        """Load JSON file as documents."""
        import json
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                data = [data]
            
            documents = []
            for item in data:
                if text_field in item:
                    metadata = {}
                    if metadata_fields:
                        for field in metadata_fields:
                            if field in item:
                                metadata[field] = item[field]
                    metadata["source"] = filepath
                    metadata["type"] = "json"
                    
                    documents.append({
                        "text": item[text_field],
                        "metadata": metadata
                    })
            
            logger.info(f"Loaded {len(documents)} documents from {filepath}")
            return documents
        
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading JSON {filepath}: {e}")
            return []
    
    @staticmethod
    def load_text(filepath: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Load plain text file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata["source"] = filepath
            doc_metadata["type"] = "text"
            
            logger.info(f"Loaded text document from {filepath}")
            return [{
                "text": content,
                "metadata": doc_metadata
            }]
        
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            return []
