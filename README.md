# SupportMind: Agentic AI Customer Support System 🤖

SupportMind is an autonomous AI customer support system that implements a self-correcting agentic loop with advanced retrieval strategies and quantitative performance evaluation.

---

## ✨ Key Features

SupportMind provides a robust set of capabilities for autonomous customer support:

1.  **Hybrid Search Architecture:** Combines dense vector embeddings (ChromaDB) with keyword-based sparse retrieval (BM25) for high-accuracy semantic and keyword-based lookup.
2.  **Semantic Re-ranking:** Integrates a Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`) to re-score documents, ensuring the most relevant context is prioritized.
3.  **Agentic Tool Orchestration:** An autonomous agent powered by a ReAct loop that can query the knowledge base, identify order statuses via mock APIs, and create CRM tickets.
4.  **Confidence-Based Escalation:** Features a self-evaluation mechanism that automatically triggers a human-handoff fallback if the agent's confidence falls below the 80% threshold.
5.  **Automated Evaluation Pipeline:** A built-in benchmarking suite using **RAGAS** to quantify Faithfulness, Answer Relevancy, and Context Recall.

---

## 🛠️ Technology Stack

*   **Core Logic:** Python 3.13+
*   **LLM Provider:** Google Gemini (2.0-Flash)
*   **Orchestration:** Custom ReAct Agent Loop
*   **Vector Store:** ChromaDB (Local Persisted)
*   **Embeddings:** `all-MiniLM-L6-v2` (Sentence-Transformers)
*   **Re-ranking:** Cross-Encoder (`ms-marco`)
*   **Evaluation:** RAGAS (LLM-based metrics)
*   **Metrics:** Custom Token and Latency trackers

---

## 🏗️ Project Architecture

```text
support-mind/
├── agents/             # The "Brain"
│   ├── agent.py        # Central Agent Logic & ReAct Loop
│   ├── tools.py        # Mock CRM/Order tools (API simulations)
│   ├── vector_db.py    # Hybrid Vector Store + BM25 implementation
│   └── memory.py       # Conversation management & short-term buffer
├── ingestion/          # Data Pipeline
│   └── indexer.py      # Semantic chunking & batch ingestion 
├── eval/               # The "Ops" Layer
│   ├── ragas_eval.py   # RAGAS metrics implementation
│   └── run_eval.py     # Batch benchmark runner
├── mock_data/          # Synthetic business data (Tickets, FAQs, MD)
├── utils/              # Config, Logging, and Performance Metrics
└── main.py             # Interactive CLI Entry point
```

---

## 🚀 Getting Started

### 1. Installation
```powershell
# Set up a venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file from the example:
```powershell
cp .env.example .env
```
*Add your `GEMINI_API_KEY` to the `.env` file.*

### 3. Running the System

**A. Interactive CLI (For Testing):**
```powershell
python main.py
```

**B. API Service (For Integration):**
```powershell
python api.py
```
*Access the interactive API documentation at **http://127.0.0.1:8000/docs***

**C. Run Performance Benchmarks:**
```powershell
python -m eval.run_eval --limit 5
```

---

## 🐳 Containerization (Docker)

To deploy the system as a portable microservice:

1. **Build the Image:**
```powershell
docker build -t support-mind-agent .
```

2. **Run the Container:**
```powershell
docker run -p 8080:8001 --env-file .env support-mind-agent
```
The API will be available at `http://localhost:8080/docs`.

---

## 📊 Monitoring & Reliability

*   **Cost & Usage Tracking:** Monitors estimated monthly costs and token consumption per interaction.
*   **Safety Guardrails:** Prevents hallucinations through confidence-threshold monitoring and automated escalation.
*   **Data-Driven Optimization:** The evaluation pipeline allows for fine-tuning RAG parameters (K-value, re-ranker) based on quantitative performance metrics.
