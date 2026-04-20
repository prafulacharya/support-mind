# Deployment & Setup Guide

## Quick Start (5 minutes)

### 1. Clone and Setup

```bash
# Clone repository (if using git)
git clone https://github.com/yourusername/support-mind.git
cd support-mind

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Claude API key
# ANTHROPIC_API_KEY=sk-...
# LANGSMITH_API_KEY=your-key  (optional, for tracing)
```

### 3. Initialize Knowledge Base

```bash
# Index all documents
python -c "from ingestion.indexer import index_knowledge_base; index_knowledge_base()"
```

Output should show:
```
Indexed 25 documents into ChromaDB
```

### 4. Run Interactive Demo

```bash
python main.py
```

You'll see:
```
======================================================================
Welcome to SupportMind - Agentic AI Customer Support System
======================================================================

📚 Initializing Vector Database...
   ✓ Knowledge base loaded: 25 documents
   ✓ Embedding model: all-MiniLM-L6-v2

🤖 Initializing Support Agent...
   ✓ Agent ready

Type 'quit' to exit, 'reset' to start new conversation, 'help' for commands
```

Try these queries:
- "How do I reset my password?"
- "What is your refund policy?"
- "Track my order ORD-001"
- "I need help with payment"

---

## Evaluation Notebook

### Run RAGAS Evaluation

```bash
# Start Jupyter
jupyter notebook

# Open eval/eval_notebook.ipynb
```

The notebook will:
1. Load test dataset (20 Q&A pairs)
2. Run baseline pipeline evaluation
3. Run optimized pipeline evaluation
4. Compare RAGAS scores
5. Generate visualizations

Expected output:
```
RAGAS Evaluation Results: Baseline vs Optimized Pipeline
================================================================
Metric              Baseline    Optimized   Improvement
Faithfulness        0.62        0.92        +48%
Answer Relevancy    0.81        0.89        +10%
Context Precision   0.62        0.86        +39%
Context Recall      0.58        0.84        +45%
Average             0.66        0.88        +33%
```

---

## Project Structure

```
support-mind/
├── README.md                      # Project overview
├── ARCHITECTURE.md                # Design decisions (key doc!)
├── requirements.txt               # Python dependencies
├── .env.example                   # Config template
│
├── agents/                        # Core AI agent
│   ├── agent.py                   # LangGraph agent
│   ├── vector_db.py               # ChromaDB wrapper
│   ├── rag_pipeline.py            # Chunking & embedding
│   ├── memory.py                  # Conversation memory
│   └── tools.py                   # Tool definitions
│
├── ingestion/                     # Data ingestion
│   └── indexer.py                 # Index documents
│
├── eval/                          # Evaluation
│   ├── ragas_eval.py              # RAGAS metrics
│   ├── test_dataset.json          # 20 Q&A pairs
│   └── eval_notebook.ipynb        # Evaluation dashboard
│
├── mock_data/                     # Test data
│   ├── faqs.md                    # Sample FAQs
│   ├── docs.md                    # Product docs
│   └── past_tickets.json          # Sample tickets
│
├── utils/                         # Utilities
│   ├── config.py                  # Configuration
│   ├── logging.py                 # Logging setup
│   ├── metrics.py                 # Cost tracking
│   └── production_patterns.py      # Production features
│
├── db/                            # Vector DB storage
│   └── chroma_db/                 # ChromaDB data (auto-created)
│
└── main.py                        # Entry point
```

---

## Configuration

### Key Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-...

# Optional - LangSmith tracing
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=support-mind

# Database
CHROMA_DB_PATH=./db/chroma_db

# RAG Pipeline
CHUNK_SIZE=512                    # tokens per chunk
CHUNK_OVERLAP=50                  # percent overlap
EMBEDDING_MODEL=all-MiniLM-L6-v2  # embedding model
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Agent
RETRIEVAL_TOP_K=5                 # documents to retrieve
RERANK_TOP_K=10                   # documents to rerank
CONFIDENCE_THRESHOLD=0.75         # escalate if below

# Logging
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR
TRACE_ENABLED=true                # enable LangSmith tracing
```

### Adding Custom Knowledge Base

1. **Add markdown docs:**
   ```bash
   # Create docs/company_faq.md
   # Contents: Markdown with # headers for sections
   
   # Update ingestion/indexer.py:
   indexer.index_markdown_file("docs/company_faq.md", category="faq")
   ```

2. **Add JSON data:**
   ```bash
   # Create docs/kb.json
   # Contents: [{"text": "...", "metadata": {...}}, ...]
   
   # Update indexing:
   indexer.index_json_file("docs/kb.json", text_field="text")
   ```

3. **Reindex:**
   ```bash
   python -c "from ingestion.indexer import index_knowledge_base; index_knowledge_base()"
   ```

---

## Testing

### Unit Tests

```bash
# Run pytest
pytest tests/

# With coverage
pytest --cov=agents tests/
```

### Integration Tests

```bash
# Run specific query
python -c "
from agents.vector_db import VectorDB
from agents.agent import SupportAgent

vector_db = VectorDB()
agent = SupportAgent(vector_db)
result = agent.process_query('How do I reset my password?')
print(result['response'])
"
```

### Performance Tests

```bash
# Measure latency
python -c "
import time
from agents.agent import SupportAgent
from agents.vector_db import VectorDB

vector_db = VectorDB()
agent = SupportAgent(vector_db)

start = time.time()
result = agent.process_query('How do I reset my password?')
latency = time.time() - start
print(f'Latency: {latency*1000:.0f}ms')
print(f'Tokens: {result[\"tokens_used\"]}')
"
```

---

## Troubleshooting

### "ANTHROPIC_API_KEY not set"
```bash
# Make sure .env file exists and has valid key
cat .env  # should show: ANTHROPIC_API_KEY=sk-...
```

### "No documents found in vector DB"
```bash
# Reindex knowledge base
python -c "from ingestion.indexer import index_knowledge_base; index_knowledge_base()"
```

### "Import errors / module not found"
```bash
# Make sure dependencies are installed
pip install -r requirements.txt

# Check Python version (3.10+)
python --version
```

### "Slow responses"
```bash
# Check if reranking is enabled (adds ~150ms)
# In config.py, see RERANK_MODEL setting

# Try with smaller knowledge base
python -c "from ingestion.indexer import index_knowledge_base; index_knowledge_base()"
```

---

## Production Deployment

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Index knowledge base
RUN python -c "from ingestion.indexer import index_knowledge_base; index_knowledge_base()"

CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t supportmind .
docker run -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY supportmind
```

### AWS Lambda

```python
import json
from agents.vector_db import VectorDB
from agents.agent import SupportAgent

vector_db = VectorDB()
agent = SupportAgent(vector_db)

def lambda_handler(event, context):
    query = event.get('query', '')
    user_id = event.get('user_id', 'anonymous')
    
    result = agent.process_query(query)
    
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

### FastAPI Server

```python
from fastapi import FastAPI
from pydantic import BaseModel
from agents.vector_db import VectorDB
from agents.agent import SupportAgent

app = FastAPI()
vector_db = VectorDB()
agent = SupportAgent(vector_db)

class Query(BaseModel):
    query: str
    user_id: str = "anonymous"

@app.post("/api/query")
async def process_query(q: Query):
    result = agent.process_query(q.query)
    return result

# Run: uvicorn api:app --reload
```

---

## Monitoring & Analytics

### View Metrics

```python
from utils.metrics import MetricsCollector

collector = MetricsCollector()
# ... run some queries ...

summary = collector.get_summary()
print(f"Total queries: {summary['total_queries']}")
print(f"Success rate: {summary['success_rate']*100:.1f}%")
print(f"Total cost: ${summary['total_cost']:.6f}")

monthly = collector.estimate_monthly_cost(queries_per_day=1000)
print(f"Estimated monthly: ${monthly['estimated_monthly_cost']}")
```

### LangSmith Traces

If LANGSMITH_API_KEY is set:
1. Go to https://smith.langchain.com
2. View traces of agent decisions
3. Debug retrieval, tool calls, reasoning

---

## Next Steps

1. **Customize Knowledge Base**
   - Replace mock_data with your company's docs
   - Add product-specific FAQs
   - Include past support tickets

2. **Fine-tune Prompts**
   - Edit `agents/agent.py` system prompt
   - Test with your specific queries
   - Measure RAGAS scores

3. **Set Up Monitoring**
   - Enable LangSmith tracing
   - Configure error alerts
   - Track RAGAS scores over time

4. **Deploy**
   - Choose hosting (AWS Lambda, Docker, Kubernetes)
   - Set up CI/CD pipeline
   - Monitor production performance

---

## Support

- **Documentation**: See README.md and ARCHITECTURE.md
- **Issues**: Check troubleshooting section above
- **Architecture questions**: See ARCHITECTURE.md
- **Code changes**: Fork and submit PR

---

**You're ready to deploy! 🚀**
