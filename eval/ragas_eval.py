"""RAGAS evaluation framework for RAG pipeline quality."""

import json
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import anthropic
from utils.logging import logger
from utils.config import Config

@dataclass
class RAGAScores:
    """RAGAS evaluation scores."""
    faithfulness: float  # Does response come from context? 0-1
    answer_relevancy: float  # Is answer relevant to question? 0-1
    context_precision: float  # Are retrieved docs relevant? 0-1
    context_recall: float  # Did we retrieve all relevant docs? 0-1
    
    @property
    def average(self) -> float:
        """Average of all metrics."""
        return (self.faithfulness + self.answer_relevancy + 
                self.context_precision + self.context_recall) / 4

class RAGASEvaluator:
    """Evaluate RAG pipeline quality using RAGAS metrics."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.results: List[Dict[str, Any]] = []
    
    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Evaluate if the answer is faithful to the provided context.
        
        Returns score from 0 to 1 where:
        - 1.0: Answer is completely grounded in context
        - 0.5: Answer is partially grounded
        - 0.0: Answer contradicts or ignores context
        """
        prompt = f"""Evaluate if this answer is faithful to the provided context.

Context:
{context}

Answer:
{answer}

Faithfulness means the answer is grounded in the context and doesn't introduce information not in the context.

Respond with just a score from 0.0 to 1.0 where:
- 1.0 = Completely faithful, all claims supported by context
- 0.75 = Mostly faithful, minor unsupported claims
- 0.5 = Partially faithful, mix of supported and unsupported claims
- 0.25 = Mostly unfaithful, mainly unsupported claims
- 0.0 = Not faithful, contradicts context or hallucinated

Score (number only):"""
        
        response = self.client.messages.create(
            model=Config.LLM_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            score = float(response.content[0].text.strip())
            return min(1.0, max(0.0, score))
        except (ValueError, IndexError):
            logger.warning("Failed to parse faithfulness score")
            return 0.5
    
    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Evaluate if answer is relevant to the question.
        
        Returns score from 0 to 1 where:
        - 1.0: Answer directly addresses the question
        - 0.5: Answer partially addresses the question
        - 0.0: Answer is not relevant
        """
        prompt = f"""Evaluate if this answer is relevant to the question.

Question:
{question}

Answer:
{answer}

Relevancy means the answer addresses the question asked.

Respond with just a score from 0.0 to 1.0 where:
- 1.0 = Directly answers the question
- 0.75 = Mostly relevant with minor tangents
- 0.5 = Partially relevant
- 0.25 = Mostly irrelevant
- 0.0 = Not relevant at all

Score (number only):"""
        
        response = self.client.messages.create(
            model=Config.LLM_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            score = float(response.content[0].text.strip())
            return min(1.0, max(0.0, score))
        except (ValueError, IndexError):
            logger.warning("Failed to parse answer relevancy score")
            return 0.5
    
    def evaluate_context_precision(self, question: str, context: List[str]) -> float:
        """
        Evaluate precision of retrieved context.
        
        Returns score from 0 to 1 where:
        - 1.0: All retrieved docs are relevant
        - 0.5: About half are relevant
        - 0.0: None are relevant
        """
        prompt = f"""Evaluate how many of these context pieces are relevant to the question.

Question:
{question}

Context pieces:
{json.dumps(context, indent=2)}

Context precision is the fraction of retrieved documents that are relevant to answering the question.

Respond with just a score from 0.0 to 1.0 where:
- 1.0 = All documents are highly relevant
- 0.75 = Most documents are relevant
- 0.5 = About half the documents are relevant
- 0.25 = Few documents are relevant
- 0.0 = No documents are relevant

Score (number only):"""
        
        response = self.client.messages.create(
            model=Config.LLM_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            score = float(response.content[0].text.strip())
            return min(1.0, max(0.0, score))
        except (ValueError, IndexError):
            logger.warning("Failed to parse context precision score")
            return 0.5
    
    def evaluate_context_recall(self, question: str, context: List[str], ground_truth: str) -> float:
        """
        Evaluate if we retrieved all relevant context.
        
        Returns score from 0 to 1 where:
        - 1.0: Retrieved all relevant information
        - 0.5: Retrieved about half
        - 0.0: Missed important information
        """
        prompt = f"""Evaluate if all relevant information was retrieved for this question.

Question:
{question}

Retrieved context:
{json.dumps(context, indent=2)}

What should have been found (ground truth):
{ground_truth}

Context recall measures whether we retrieved all the relevant information needed to answer the question.

Respond with just a score from 0.0 to 1.0 where:
- 1.0 = All relevant information was retrieved
- 0.75 = Most relevant information was retrieved
- 0.5 = About half the relevant information was retrieved
- 0.25 = Little relevant information was retrieved
- 0.0 = No relevant information was retrieved

Score (number only):"""
        
        response = self.client.messages.create(
            model=Config.LLM_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            score = float(response.content[0].text.strip())
            return min(1.0, max(0.0, score))
        except (ValueError, IndexError):
            logger.warning("Failed to parse context recall score")
            return 0.5
    
    def evaluate(self,
                question: str,
                answer: str,
                context: List[str],
                ground_truth: str = "") -> RAGAScores:
        """
        Evaluate a single QA pair comprehensively.
        
        Args:
            question: The question asked
            answer: The answer provided
            context: List of context documents retrieved
            ground_truth: What should have been found
        
        Returns:
            RAGAScores with all metrics
        """
        logger.info(f"Evaluating: {question[:50]}...")
        
        # Format context for evaluation
        context_str = "\n---\n".join(context)
        
        # Calculate each metric
        faithfulness = self.evaluate_faithfulness(answer, context_str)
        answer_relevancy = self.evaluate_answer_relevancy(question, answer)
        context_precision = self.evaluate_context_precision(question, context)
        context_recall = self.evaluate_context_recall(question, context, ground_truth)
        
        scores = RAGAScores(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall
        )
        
        # Store result
        self.results.append({
            "question": question,
            "answer": answer,
            "scores": {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "average": scores.average
            },
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"RAGAS Scores - F:{faithfulness:.2f} AR:{answer_relevancy:.2f} CP:{context_precision:.2f} CR:{context_recall:.2f}")
        
        return scores
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all evaluations."""
        if not self.results:
            return {}
        
        faithfulness_scores = [r["scores"]["faithfulness"] for r in self.results]
        answer_relevancy_scores = [r["scores"]["answer_relevancy"] for r in self.results]
        context_precision_scores = [r["scores"]["context_precision"] for r in self.results]
        context_recall_scores = [r["scores"]["context_recall"] for r in self.results]
        average_scores = [r["scores"]["average"] for r in self.results]
        
        return {
            "total_evaluations": len(self.results),
            "average_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
            "average_answer_relevancy": sum(answer_relevancy_scores) / len(answer_relevancy_scores),
            "average_context_precision": sum(context_precision_scores) / len(context_precision_scores),
            "average_context_recall": sum(context_recall_scores) / len(context_recall_scores),
            "overall_average": sum(average_scores) / len(average_scores),
            "evaluations": self.results
        }
    
    def save_results(self, filepath: str) -> None:
        """Save evaluation results to JSON."""
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved evaluation results to {filepath}")
