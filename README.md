# SupportMind: Production-Grade Agentic RAG System 🤖

SupportMind is an advanced, autonomous AI customer support system designed to demonstrate senior-level AI engineering competencies. It goes beyond simple "Chat-over-PDF" tutorials by implementing a robust, self-correcting agentic loop with complex retrieval strategies and quantitative performance evaluation.

## 🌟 Why this is "Production-Grade"

Most AI projects stop at basic RAG. SupportMind implements the "last mile" features required for real-world reliability:

1.  **Hybrid Search Architecture:** Combines dense vector embeddings (ChromaDB) with keyword-based sparse retrieval (BM25) to handle both semantic questions and specific keyword lookups (like order IDs).
2.  **Semantic Re-ranking:** Uses a Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`) to re-score retrieved documents, significantly improving context precision over raw vector search.
3.  **Agentic Tool Orchestration:** The agent isn't just a chatbot; it's a decision-maker. It can choose to search the knowledge base, check order statuses via mock APIs, or create support tickets in a CRM.
4.  **Self-Evaluation & Escalation:** Implements a confidence-scoring mechanism. If the agent's internal confidence drops below a threshold (80%), it autonomously triggers the `escalate_to_human` fallback to prevent hallucinations.
5.  **Quantitative Observability (RAGAS):** Includes a dedicated benchmarking suite that evaluates responses based on **Faithfulness**, **Answer Relevancy**, and **Context Recall** using Gemini as an LLM-judge.

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
**A. First-time Ingestion:**
The system will automatically index `/mock_data` on the first run of `main.py`.

**B. Launch the Agent:**
```powershell
python main.py
```

**C. Run Performance Benchmarks:**
```powershell
python -m eval.run_eval --limit 5
```

---

## 📈 Impact for Hiring Managers

*   **Cost Control:** Tracks estimated monthly costs and token usage per turn.
*   **Reliability:** Demonstrates how to handle hallucinations using threshold-based escalation.
*   **Optimization:** The RAGAS pipeline proves that the RAG parameters (K-value, re-ranker) were chosen based on data, not guesses.
