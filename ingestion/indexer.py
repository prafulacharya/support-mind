"""Document ingestion and indexing."""

import os
from typing import List, Dict, Any
from agents.rag_pipeline import RAGPipeline, DocumentLoader
from agents.vector_db import VectorDB
from utils.logging import logger

class DocumentIndexer:
    """Index documents into the vector database."""
    
    def __init__(self, vector_db: VectorDB):
        """Initialize the indexer."""
        self.vector_db = vector_db
        self.rag_pipeline = RAGPipeline()
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index documents into the vector database.
        
        Args:
            documents: List of documents with 'text' and optional 'metadata'
        """
        # Chunk documents
        chunks = self.rag_pipeline.process_documents(documents)
        
        # Add to vector database
        self.vector_db.add_documents(chunks)
        
        logger.info(f"Indexed {len(chunks)} chunks from {len(documents)} documents")
    
    def index_markdown_file(self, filepath: str, category: str = "faq") -> None:
        """Index a markdown file."""
        documents = DocumentLoader.load_markdown(filepath, {"category": category})
        self.index_documents(documents)
    
    def index_json_file(self, filepath: str, text_field: str = "text", category: str = "general") -> None:
        """Index a JSON file."""
        documents = DocumentLoader.load_json(filepath, text_field)
        # Add category to metadata
        for doc in documents:
            if "metadata" not in doc:
                doc["metadata"] = {}
            doc["metadata"]["category"] = category
        
        self.index_documents(documents)

def index_knowledge_base():
    """Index all knowledge base files."""
    # Initialize vector DB
    vector_db = VectorDB()
    indexer = DocumentIndexer(vector_db)
    
    # Define paths to knowledge base files
    mock_data_dir = "mock_data"
    
    # Index FAQs
    faqs_path = os.path.join(mock_data_dir, "faqs.md")
    if os.path.exists(faqs_path):
        logger.info(f"Indexing {faqs_path}...")
        indexer.index_markdown_file(faqs_path, category="faq")
    
    # Index product docs
    docs_path = os.path.join(mock_data_dir, "docs.md")
    if os.path.exists(docs_path):
        logger.info(f"Indexing {docs_path}...")
        indexer.index_markdown_file(docs_path, category="documentation")
    
    # Index past tickets
    tickets_path = os.path.join(mock_data_dir, "past_tickets.json")
    if os.path.exists(tickets_path):
        logger.info(f"Indexing {tickets_path}...")
        indexer.index_json_file(tickets_path, text_field="resolution", category="resolved_tickets")
    
    # Print stats
    stats = vector_db.get_stats()
    logger.info(f"Knowledge base indexed: {stats['document_count']} documents")
    print(f"Indexed {stats['document_count']} documents into ChromaDB")

if __name__ == "__main__":
    index_knowledge_base()
