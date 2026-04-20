# ChromaDB Storage Directory

This directory contains the persistent ChromaDB vector database storage.

The database is automatically initialized when you first run the application.

To clear the database, simply delete the files in this directory and reindex:
```bash
python -c "from ingestion.indexer import index_knowledge_base; index_knowledge_base()"
```
