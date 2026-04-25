"""RAGAS evaluation framework for RAG pipeline quality."""

import json
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import google.generativeai as genai
from utils.logging import logger
from utils.config import Config
import re

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
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(Config.LLM_MODEL)
        self.results: List[Dict[str, Any]] = []
    
    def _extract_score(self, text: str) -> float:
        """Extract a float score from the LLM response."""
        match = re.search(r'([0-9]?\.[0-9]+|[0-9])', text)
        if match:
            return min(1.0, max(0.0, float(match.group(1))))
        return 0.5

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        prompt = f"""Evaluate if this answer is faithful to the provided context.

Context:
{context}

Answer:
{answer}

Faithfulness means the answer is grounded in the context and doesn't introduce information not in the context.

Respond with ONLY a decimal score from 0.0 to 1.0 where:
1.0 = Completely faithful, all claims supported by context
0.0 = Not faithful, contradicts context or hallucinated

Score:"""
        
        try:
            response = self.model.generate_content(prompt)
            return self._extract_score(response.text)
        except Exception as e:
            logger.warning(f"Failed to score faithfulness: {e}")
            return 0.5
    
    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        prompt = f"""Evaluate if this answer is relevant to the question.

Question:
{question}

Answer:
{answer}

Relevancy means the answer addresses the question asked.

Respond with ONLY a decimal score from 0.0 to 1.0 where:
1.0 = Directly answers the question
0.0 = Not relevant at all

Score:"""
        
        try:
            response = self.model.generate_content(prompt)
            return self._extract_score(response.text)
        except Exception as e:
            logger.warning(f"Failed to score answer relevancy: {e}")
            return 0.5
    
    def evaluate_context_precision(self, question: str, context: List[str]) -> float:
        prompt = f"""Evaluate how many of these context pieces are relevant to the question.

Question:
{question}

Context pieces:
{json.dumps(context, indent=2)}

Context precision is the fraction of retrieved documents that are relevant to answering the question.

Respond with ONLY a decimal score from 0.0 to 1.0 where:
1.0 = All documents are highly relevant
0.0 = No documents are relevant

Score:"""
        
        try:
            response = self.model.generate_content(prompt)
            return self._extract_score(response.text)
        except Exception as e:
            logger.warning(f"Failed to score context precision: {e}")
            return 0.5
    
    def evaluate_context_recall(self, question: str, context: List[str], ground_truth: str) -> float:
        prompt = f"""Evaluate if all relevant information was retrieved for this question.

Question:
{question}

Retrieved context:
{json.dumps(context, indent=2)}

What should have been found (ground truth):
{ground_truth}

Context recall measures whether we retrieved all the relevant information needed to answer the question.

Respond with ONLY a decimal score from 0.0 to 1.0 where:
1.0 = All relevant information was retrieved
0.0 = No relevant information was retrieved

Score:"""
        
        try:
            response = self.model.generate_content(prompt)
            return self._extract_score(response.text)
        except Exception as e:
            logger.warning(f"Failed to score context recall: {e}")
            return 0.5
    
    def evaluate(self,
                question: str,
                answer: str,
                context: List[str],
                ground_truth: str = "") -> RAGAScores:
        logger.info(f"Evaluating: {question[:50]}...")
        
        context_str = "\n---\n".join(context)
        
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
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved evaluation results to {filepath}")
