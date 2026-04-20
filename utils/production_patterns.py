"""Production patterns and best practices for SupportMind."""

from typing import Dict, Any, Optional, Generator
import json
import time
from datetime import datetime
from agents.agent import SupportAgent
from utils.logging import logger

class StreamingResponse:
    """Streaming response handler for real-time updates."""
    
    def __init__(self, agent: SupportAgent):
        self.agent = agent
    
    def stream_query(self, query: str) -> Generator[str, None, None]:
        """
        Stream agent thinking and response.
        
        Yields:
            Chunks of text as they're generated
        """
        # Simulate streaming by yielding chunks
        yield "[THINKING] Analyzing your query...\n"
        yield "[THOUGHT] I need to retrieve relevant documents.\n"
        
        # Get retrieval results
        docs = self.agent.vector_db.hybrid_retrieve(query, top_k=5, rerank=True)
        yield f"[RETRIEVED] Found {len(docs)} relevant documents.\n"
        
        # Process query
        result = self.agent.process_query(query)
        
        yield f"[RESPONSE] {result['response']}\n"
        yield f"[CONFIDENCE] {result['confidence']*100:.0f}%\n"
        yield f"[STATS] Tokens: {result['tokens_used']}, Latency: {result['latency_ms']:.0f}ms\n"

class FallbackHandler:
    """Handle failures gracefully with fallbacks."""
    
    @staticmethod
    def fallback_keyword_search(query: str, vector_db) -> str:
        """Fallback to keyword search if semantic search fails."""
        logger.warning(f"Falling back to keyword search for: {query}")
        
        # BM25 keyword search
        results = vector_db.retrieve(query, top_k=3, hybrid=False)
        
        if results:
            context = "\n".join([r["text"] for r in results])
            return f"I found these potentially relevant articles:\n{context}"
        else:
            return "I couldn't find relevant information. Please contact support."
    
    @staticmethod
    def escalate_gracefully(ticket_id: str, reason: str) -> Dict[str, Any]:
        """Escalate to human support with proper context."""
        logger.warning(f"Escalating ticket {ticket_id}: {reason}")
        
        return {
            "status": "escalated",
            "ticket_id": ticket_id,
            "message": f"Your issue has been escalated to our support team. Ticket ID: {ticket_id}",
            "estimated_response_time": "30 minutes",
            "next_steps": [
                "A human agent will review your case",
                "You'll receive an email update",
                "You can check status anytime at supportmind.com/tickets/{ticket_id}"
            ]
        }

class RateLimiter:
    """Rate limiting for production stability."""
    
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.requests: Dict[str, list] = {}
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        
        if user_id not in self.requests:
            self.requests[user_id] = []
        
        # Clean old requests (older than 1 minute)
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id]
            if now - req_time < 60
        ]
        
        # Check limit
        if len(self.requests[user_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[user_id].append(now)
        return True

class CostEstimator:
    """Estimate and track query costs."""
    
    # Claude Sonnet 3.5 pricing (as of 2024)
    INPUT_COST_PER_1M_TOKENS = 3.0
    OUTPUT_COST_PER_1M_TOKENS = 15.0
    
    @staticmethod
    def estimate_query_cost(input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a single query."""
        input_cost = (input_tokens / 1_000_000) * CostEstimator.INPUT_COST_PER_1M_TOKENS
        output_cost = (output_tokens / 1_000_000) * CostEstimator.OUTPUT_COST_PER_1M_TOKENS
        return input_cost + output_cost
    
    @staticmethod
    def estimate_monthly_cost(queries_per_day: int, avg_input_tokens: int, avg_output_tokens: int) -> Dict[str, Any]:
        """Estimate monthly operational cost."""
        cost_per_query = CostEstimator.estimate_query_cost(avg_input_tokens, avg_output_tokens)
        monthly_queries = queries_per_day * 30
        
        return {
            "queries_per_day": queries_per_day,
            "queries_per_month": monthly_queries,
            "cost_per_query": round(cost_per_query, 6),
            "monthly_cost": round(cost_per_query * monthly_queries, 2),
            "yearly_cost": round(cost_per_query * monthly_queries * 12, 2),
            "breakdown": {
                "average_input_tokens": avg_input_tokens,
                "average_output_tokens": avg_output_tokens,
                "input_cost_per_query": round((avg_input_tokens / 1_000_000) * CostEstimator.INPUT_COST_PER_1M_TOKENS, 6),
                "output_cost_per_query": round((avg_output_tokens / 1_000_000) * CostEstimator.OUTPUT_COST_PER_1M_TOKENS, 6)
            }
        }

class HallucinationDetector:
    """Detect and prevent hallucinations."""
    
    FORBIDDEN_PATTERNS = {
        "order_number": r"ORD-\d+",  # Don't make up order numbers
        "price": r"\$\d+(\.\d+)?",   # Don't make up prices
        "date": r"\d{4}-\d{2}-\d{2}",  # Don't make up dates
    }
    
    @staticmethod
    def check_for_hallucinations(response: str, context: str) -> Dict[str, Any]:
        """Check if response contains hallucinated information."""
        import re
        
        hallucinations = []
        
        for pattern_type, pattern in HallucinationDetector.FORBIDDEN_PATTERNS.items():
            # Find patterns in response
            response_matches = re.findall(pattern, response)
            
            # Check if they're in context
            for match in response_matches:
                if match not in context:
                    hallucinations.append({
                        "type": pattern_type,
                        "value": match,
                        "severity": "high"
                    })
        
        return {
            "has_hallucinations": len(hallucinations) > 0,
            "hallucinations": hallucinations,
            "score": 1.0 - (len(hallucinations) * 0.1)  # Deduct 0.1 for each hallucination
        }

class PromptCaching:
    """Cache responses to identical queries."""
    
    def __init__(self, max_cache_size: int = 100):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_cache_size
        self.hits = 0
        self.misses = 0
    
    def get(self, query: str) -> Optional[Any]:
        """Get cached response."""
        if query in self.cache:
            self.hits += 1
            return self.cache[query]
        self.misses += 1
        return None
    
    def set(self, query: str, response: Any) -> None:
        """Cache a response."""
        # Simple LRU by removing oldest if cache is full
        if len(self.cache) >= self.max_size:
            # Remove first item
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[query] = response
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "total_requests": total,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache)
        }

class ProductionAgent:
    """Production-ready agent with all patterns."""
    
    def __init__(self, agent: SupportAgent):
        self.agent = agent
        self.rate_limiter = RateLimiter(max_requests_per_minute=100)
        self.cost_estimator = CostEstimator()
        self.hallucination_detector = HallucinationDetector()
        self.cache = PromptCaching(max_cache_size=1000)
        self.streaming = StreamingResponse(agent)
        self.query_log: list = []
    
    def process_query_production(self, user_id: str, query: str) -> Dict[str, Any]:
        """
        Process query with all production patterns.
        
        Args:
            user_id: User identifier for rate limiting
            query: User query
        
        Returns:
            Response with metadata
        """
        # Rate limiting
        if not self.rate_limiter.is_allowed(user_id):
            return {
                "error": "Rate limit exceeded",
                "status": 429,
                "message": "Too many requests. Please try again in a moment."
            }
        
        # Cache check
        cached = self.cache.get(query)
        if cached:
            cached["from_cache"] = True
            return cached
        
        # Process query
        start_time = time.time()
        result = self.agent.process_query(query)
        latency = time.time() - start_time
        
        # Hallucination detection
        context = " ".join([result.get('response', '')])
        hallucination_check = self.hallucination_detector.check_for_hallucinations(
            result['response'],
            context
        )
        
        # If hallucinations detected, escalate
        if hallucination_check['has_hallucinations']:
            logger.warning(f"Hallucinations detected: {hallucination_check['hallucinations']}")
            result['escalated'] = True
            result['escalation_reason'] = "System detected potential hallucinations"
        
        # Cost estimation
        cost = self.cost_estimator.estimate_query_cost(
            result.get('tokens_used', 100) * 0.7,  # Rough estimate
            result.get('tokens_used', 100) * 0.3
        )
        
        # Prepare response
        response = {
            **result,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "latency_ms": latency * 1000,
            "estimated_cost": cost,
            "from_cache": False,
            "hallucination_check": hallucination_check
        }
        
        # Cache it
        self.cache.set(query, response)
        
        # Log query
        self.query_log.append(response)
        
        return response
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics for all processed queries."""
        if not self.query_log:
            return {}
        
        return {
            "total_queries": len(self.query_log),
            "cache_stats": self.cache.get_stats(),
            "avg_latency_ms": sum(q['latency_ms'] for q in self.query_log) / len(self.query_log),
            "total_cost": sum(q.get('estimated_cost', 0) for q in self.query_log),
            "escalation_rate": sum(1 for q in self.query_log if q.get('escalated')) / len(self.query_log),
            "hallucination_rate": sum(1 for q in self.query_log if q.get('hallucination_check', {}).get('has_hallucinations')) / len(self.query_log)
        }

def setup_production_monitoring():
    """Setup monitoring and alerting for production."""
    logger.info("Production monitoring initialized")
    
    return {
        "metrics_to_monitor": [
            "Query latency (target: < 2s)",
            "Token usage per query",
            "Escalation rate (target: < 10%)",
            "Hallucination rate (target: < 1%)",
            "Cache hit rate (target: > 20%)",
            "Cost per query"
        ],
        "alert_thresholds": {
            "latency_ms": 2000,
            "tokens_per_query": 5000,
            "escalation_rate": 0.10,
            "hallucination_rate": 0.01,
            "error_rate": 0.05
        },
        "monitoring_frequency": "every 5 minutes"
    }
