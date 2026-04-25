from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import time

from agents.vector_db import VectorDB
from agents.agent import SupportAgent
from utils.config import Config
from utils.logging import logger

# Initialize FastAPI app
app = FastAPI(
    title="SupportMind AI API",
    description="Production-grade AI Customer Support Agent API",
    version="1.0.0"
)

# Models for request/response
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default_session"

class ChatResponse(BaseModel):
    query: str
    response: str
    escalated: bool
    confidence: float
    documents_used: int
    tokens_used: int
    latency_ms: float

# Global variables for components
vector_db = None
agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global vector_db, agent
    logger.info("Initializing Agentic System...")
    
    # Initialize Vector DB
    vector_db = VectorDB()
    
    # Check if DB is empty and would need ingestion (in a real prod app, you'd handle this via a separate job)
    stats = vector_db.get_stats()
    if stats["document_count"] == 0:
        logger.warning("Vector DB is empty. Please run ingestion before using the API.")
    
    # Initialize Agent
    agent = SupportAgent(vector_db)
    logger.info("SupportMind API is ready.")

@app.get("/")
async def root():
    return {"message": "SupportMind AI Agent API is online", "status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a customer support query.
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Process query through the agentic loop
        result = agent.process_query(request.query)
        
        return ChatResponse(
            query=result["query"],
            response=result["response"],
            escalated=result["escalated"],
            confidence=result["confidence"],
            documents_used=result["documents_used"],
            tokens_used=result["tokens_used"],
            latency_ms=result["latency_ms"]
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_conversation():
    """Reset the current conversation state."""
    if agent:
        agent.reset_conversation()
        return {"message": "Conversation reset successfully"}
    return {"message": "Agent not initialized"}

if __name__ == "__main__":
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api:app", host=host, port=port, reload=True)
