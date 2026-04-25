import os
import json
import argparse
from tqdm import tqdm
from agents.vector_db import VectorDB
from agents.agent import SupportAgent
from eval.ragas_eval import RAGASEvaluator
from utils.logging import logger

def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on SupportMind")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries to evaluate")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file for evaluation results")
    args = parser.parse_args()

    print("Initializing Vector DB and Agent...")
    vector_db = VectorDB()
    agent = SupportAgent(vector_db)
    evaluator = RAGASEvaluator()

    dataset_path = "eval/test_dataset.json"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.limit:
        dataset = dataset[:args.limit]

    print(f"\nRunning evaluation on {len(dataset)} items...\n")
    
    for item in tqdm(dataset):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        # Reset memory for independent testing
        agent.reset_conversation()
        
        # We need to capture context, our agent currently retrieves inside process_query
        # but we can get it manually for evaluation purposes
        context_docs = agent.retrieve_context(question)
        context_strs = [doc["text"] for doc in context_docs]
        
        # Run agent
        result = agent.process_query(question, max_iterations=3)
        answer = result["response"]
        
        # Evaluate
        evaluator.evaluate(
            question=question,
            answer=answer,
            context=context_strs,
            ground_truth=ground_truth
        )

    # Output results
    summary = evaluator.get_summary()
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Evaluations: {summary['total_evaluations']}")
    print(f"Average Faithfulness:      {summary['average_faithfulness']:.2f}")
    print(f"Average Answer Relevancy:  {summary['average_answer_relevancy']:.2f}")
    print(f"Average Context Precision: {summary['average_context_precision']:.2f}")
    print(f"Average Context Recall:    {summary['average_context_recall']:.2f}")
    print(f"OVERALL AVERAGE:           {summary['overall_average']:.2f}")
    print("="*50)

    # Save to file
    evaluator.save_results(args.output)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
