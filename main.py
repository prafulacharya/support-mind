"""Main entry point for SupportMind demo."""

import time
from agents.vector_db import VectorDB
from agents.agent import SupportAgent
from ingestion.indexer import index_knowledge_base
from utils.config import Config
from utils.logging import logger
from utils.metrics import MetricsCollector, TokenMetrics, LatencyMetrics

def main():
    """Run the interactive support agent demo."""
    
    print("\n" + "="*70)
    print("Welcome to SupportMind - Agentic AI Customer Support System")
    print("="*70 + "\n")
    
    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Please set ANTHROPIC_API_KEY in .env file")
        return
    
    # Initialize vector DB
    print("📚 Initializing Vector Database...")
    vector_db = VectorDB()
    stats = vector_db.get_stats()
    
    if stats["document_count"] == 0:
        print("   ℹ️  No documents found. Indexing knowledge base...")
        index_knowledge_base()
        stats = vector_db.get_stats()
    
    print(f"   ✓ Knowledge base loaded: {stats['document_count']} documents")
    print(f"   ✓ Embedding model: {stats['embedding_model']}\n")
    
    # Initialize agent
    print("🤖 Initializing Support Agent...")
    agent = SupportAgent(vector_db)
    print("   ✓ Agent ready\n")
    
    # Initialize metrics
    metrics_collector = MetricsCollector()
    
    # Interactive loop
    print("Type 'quit' to exit, 'reset' to start new conversation, 'help' for commands\n")
    print("-" * 70)
    
    query_count = 0
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                print("\n👋 Thank you for using SupportMind!")
                print_final_metrics(metrics_collector)
                break
            
            if user_input.lower() == "reset":
                print("🔄 Resetting conversation...")
                agent.reset_conversation()
                print("✓ Conversation reset. Starting fresh.\n")
                continue
            
            if user_input.lower() == "help":
                print_help()
                continue
            
            if user_input.lower() == "history":
                print_conversation_history(agent)
                continue
            
            # Process query
            print("\n⏳ Processing your query...")
            start_time = time.time()
            
            result = agent.process_query(user_input)
            
            latency = time.time() - start_time
            query_count += 1
            
            # Display response
            print(f"\n🤖 Agent: {result['response']}\n")
            
            # Display metrics
            if result.get("escalated"):
                print("⚠️  [Escalated to human support]")
            else:
                confidence_percent = result.get("confidence", 0) * 100
                confidence_bar = create_confidence_bar(confidence_percent)
                print(f"   Confidence: {confidence_bar} {confidence_percent:.0f}%")
            
            print(f"   Documents used: {result.get('documents_used', 0)}")
            print(f"   Tokens used: {result.get('tokens_used', 0)}")
            print(f"   Latency: {latency:.2f}s")
            
            # Collect metrics (mock token counts for demo)
            token_metrics = TokenMetrics(
                input_tokens=result.get('tokens_used', 100),
                output_tokens=result.get('tokens_used', 50) // 2
            )
            latency_metrics = LatencyMetrics(llm_ms=latency * 1000)
            
            metrics_collector.add_query(
                query=user_input,
                token_metrics=token_metrics,
                latency_metrics=latency_metrics,
                success=not result.get("escalated", False)
            )
            
            print("-" * 70)
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Thank you for using SupportMind!")
            print_final_metrics(metrics_collector)
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'quit' to exit.\n")

def print_help():
    """Print help information."""
    print("\n" + "="*70)
    print("Commands:")
    print("  'quit'    - Exit the application")
    print("  'reset'   - Start a new conversation")
    print("  'history' - View conversation history")
    print("  'help'    - Show this help message")
    print("="*70 + "\n")

def print_conversation_history(agent: SupportAgent):
    """Print the conversation history."""
    history = agent.get_conversation_history()
    print("\n" + "="*70)
    print("Conversation History:")
    print("="*70)
    for i, msg in enumerate(history, 1):
        role = "👤 You" if msg["role"] == "user" else "🤖 Agent"
        print(f"\n[{i}] {role}:")
        print(f"    {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
    print("\n" + "="*70 + "\n")

def create_confidence_bar(confidence_percent: float) -> str:
    """Create a visual confidence bar."""
    filled = int(confidence_percent / 10)
    empty = 10 - filled
    return f"[{'█' * filled}{'░' * empty}]"

def print_final_metrics(metrics_collector: MetricsCollector):
    """Print final statistics."""
    summary = metrics_collector.get_summary()
    
    if not summary:
        return
    
    print("\n" + "="*70)
    print("Session Statistics:")
    print("="*70)
    print(f"Total queries: {summary['total_queries']}")
    print(f"Successful: {summary['successful']}")
    print(f"Success rate: {summary['success_rate']*100:.1f}%")
    print(f"Total tokens: {summary['total_tokens']}")
    print(f"Total cost: ${summary['total_cost']:.6f}")
    print(f"Average latency: {summary['avg_latency_ms']:.0f}ms")
    
    # Monthly estimate
    monthly = metrics_collector.estimate_monthly_cost()
    if monthly:
        print(f"\nEstimated monthly cost (1000 queries/day):")
        print(f"  ${monthly['estimated_monthly_cost']:.2f}/month")
        print(f"  ${monthly['estimated_yearly_cost']:.2f}/year")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
