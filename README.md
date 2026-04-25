# SupportMind: Enterprise-Grade Agentic AI Customer Support

SupportMind is an autonomous AI customer support microservice that implements a self-correcting agentic loop with advanced hybrid retrieval strategies and quantitative performance evaluation.

---

## Technical Highlights

*   **Hybrid Search Architecture**: Merges dense vector embeddings (ChromaDB) with keyword-based sparse retrieval (BM25) for sub-millisecond, high-accuracy lookup.
*   **Production-Ready API**: Wrapped in a high-performance FastAPI server with automated Swagger documentation and stateless session management.
*   **Agentic Orchestration**: Powered by an autonomous ReAct loop that executes complex reasoning, tool usage (CRM/Order APIs), and self-correction.
*   **Containerized Deployment**: Fully Dockerized with multi-stage build optimization for seamless cloud deployment (AWS, GCP, Azure).
*   **Quantitative Observability**: Built-in RAGAS evaluation pipeline to monitor Faithfulness, Relevancy, and Recall metrics.

---

## Project Structure

```text
support-mind/
├── agents/             # Core AI Intelligence
│   ├── agent.py        # Central Agent Logic & ReAct reasoning loop
│   ├── tools.py        # Validated CRM/Order tool definitions
│   ├── vector_db.py    # Hybrid Vector Store (ChromaDB + BM25)
│   └── memory.py       # Stateless conversation context management
├── ingestion/          # Data Engineering Pipeline
│   └── indexer.py      # Automated semantic chunking & indexing
├── eval/               # AI Observability & Performance
│   ├── ragas_eval.py   # RAGAS metric implementations
│   └── run_eval.py     # Batch performance benchmarking suite
├── utils/              # System Utilities
│   ├── config.py       # Scalable environment configuration
│   ├── logging.py      # Structured canonical logging
│   └── metrics.py      # Token usage and latency tracking
├── api.py              # FastAPI Microservice Entry point
├── main.py             # Internal Developer CLI
├── Dockerfile          # Containerization specification
├── .dockerignore       # Build context optimization
└── requirements.txt    # Dependency management
```

---

## Getting Started

### 1. Installation & Environment
```powershell
# Initialize virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install core dependencies
pip install -r requirements.txt
```

### 2. Configuration
Define your environment variables in a `.env` file:
```env
GEMINI_API_KEY=your_api_key_here
LLM_MODEL=gemini-2.5-flash-lite
```

### 3. Execution Modes

**A. API Microservice (Recommended for Production):**
```powershell
python api.py
```
*Access interactive documentation at http://127.0.0.1:8000/docs*

**B. Interactive CLI (Recommended for Development):**
```powershell
python main.py
```

---

## Deployment (Docker)

To ship SupportMind as a portable, isolated microservice:

1. **Build the Production Image:**
```powershell
docker build -t support-mind-agent .
```

2. **Run the Optimized Container:**
```powershell
docker run -p 8080:8000 --env-file .env support-mind-agent
```
The service will be reachable at `http://localhost:8080/docs`.

---

## Reliability & Governance

*   **Confidence Guardrails**: Automated escalation to human-handoff if agent confidence falls below established thresholds.
*   **Cost Management**: Granular token consumption tracking to prevent budget overruns in high-volume environments.
*   **Stateless Scaling**: Designed to run in distributed environments without context leakage across sessions.
