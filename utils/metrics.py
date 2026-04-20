from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

@dataclass
class TokenMetrics:
    """Token usage for a single request."""
    input_tokens: int
    output_tokens: int
    model: str = "claude-3-5-sonnet-20241022"
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    @property
    def input_cost(self) -> float:
        """Cost for input tokens (Claude Sonnet: $3 per 1M tokens)."""
        return (self.input_tokens / 1_000_000) * 3.0
    
    @property
    def output_cost(self) -> float:
        """Cost for output tokens (Claude Sonnet: $15 per 1M tokens)."""
        return (self.output_tokens / 1_000_000) * 15.0
    
    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost

@dataclass
class LatencyMetrics:
    """Latency tracking for components."""
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    llm_ms: float = 0.0
    tool_calls_ms: float = 0.0
    
    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.reranking_ms + self.llm_ms + self.tool_calls_ms
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "retrieval_ms": self.retrieval_ms,
            "reranking_ms": self.reranking_ms,
            "llm_ms": self.llm_ms,
            "tool_calls_ms": self.tool_calls_ms,
            "total_ms": self.total_ms
        }

class MetricsCollector:
    """Collect and aggregate metrics across queries."""
    
    def __init__(self):
        self.queries: List[Dict] = []
    
    def add_query(self, 
                  query: str, 
                  token_metrics: TokenMetrics,
                  latency_metrics: LatencyMetrics,
                  success: bool = True,
                  error: str = None):
        """Record metrics for a query."""
        self.queries.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "success": success,
            "error": error,
            "tokens": {
                "input": token_metrics.input_tokens,
                "output": token_metrics.output_tokens,
                "total": token_metrics.total_tokens,
                "cost": round(token_metrics.total_cost, 6)
            },
            "latency": latency_metrics.to_dict()
        })
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.queries:
            return {}
        
        successful = [q for q in self.queries if q["success"]]
        total_cost = sum(q["tokens"]["cost"] for q in self.queries)
        avg_latency = sum(q["latency"]["total_ms"] for q in self.queries) / len(self.queries)
        
        return {
            "total_queries": len(self.queries),
            "successful": len(successful),
            "success_rate": len(successful) / len(self.queries),
            "total_tokens": sum(q["tokens"]["total"] for q in self.queries),
            "total_cost": round(total_cost, 6),
            "avg_latency_ms": round(avg_latency, 2),
            "queries_processed": self.queries
        }
    
    def estimate_monthly_cost(self, queries_per_day: int = 1000) -> Dict:
        """Estimate monthly costs based on current metrics."""
        if not self.queries:
            return {}
        
        avg_cost_per_query = sum(q["tokens"]["cost"] for q in self.queries) / len(self.queries)
        
        return {
            "queries_per_day": queries_per_day,
            "queries_per_month": queries_per_day * 30,
            "avg_cost_per_query": round(avg_cost_per_query, 6),
            "estimated_monthly_cost": round(avg_cost_per_query * queries_per_day * 30, 2),
            "estimated_yearly_cost": round(avg_cost_per_query * queries_per_day * 365, 2)
        }
