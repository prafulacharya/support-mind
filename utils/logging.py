import logging
import json
from datetime import datetime
from typing import Any, Dict
from utils.config import Config

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def log_retrieval(query: str, documents: list, scores: list) -> None:
    """Log retrieval operation with documents and scores."""
    logger.info(f"Query: {query}")
    logger.info(f"Retrieved {len(documents)} documents")
    for i, (doc, score) in enumerate(zip(documents, scores)):
        logger.debug(f"  [{i+1}] Score: {score:.4f} | {doc[:100]}...")

def log_agent_action(thought: str, action: str, tool: str, input_data: Dict[str, Any]) -> None:
    """Log agent thinking and actions."""
    logger.info(f"[THOUGHT] {thought}")
    logger.info(f"[ACTION] {action} → Tool: {tool}")
    logger.debug(f"[INPUT] {json.dumps(input_data, indent=2)}")

def log_agent_observation(observation: Any) -> None:
    """Log agent observations."""
    if isinstance(observation, dict):
        logger.info(f"[OBSERVATION] {json.dumps(observation, indent=2)}")
    else:
        logger.info(f"[OBSERVATION] {str(observation)[:200]}")

def log_metrics(metrics: Dict[str, Any]) -> None:
    """Log performance metrics."""
    logger.info("[METRICS]")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")

def log_escalation(reason: str, ticket_id: str) -> None:
    """Log escalation events."""
    logger.warning(f"[ESCALATION] Reason: {reason} | Ticket ID: {ticket_id}")

class TraceEvent:
    """Event for LangSmith tracing."""
    def __init__(self, event_type: str, name: str, data: Dict[str, Any]):
        self.event_type = event_type  # "start", "end", "error"
        self.name = name
        self.data = data
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "name": self.name,
            "timestamp": self.timestamp,
            "data": self.data
        }
