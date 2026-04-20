"""Agentic AI using LangGraph."""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import re
import anthropic
from langgraph.graph import StateGraph, END
from agents.vector_db import VectorDB
from agents.memory import ConversationManager
from agents.tools import TOOLS, execute_tool
from utils.config import Config
from utils.logging import logger
from utils.metrics import TokenMetrics, LatencyMetrics
import time

# State schema for the graph
class AgentState:
    """State maintained throughout agent execution."""
    def __init__(self):
        self.query: str = ""
        self.context: str = ""
        self.retrieved_docs: List[Dict[str, Any]] = []
        self.thoughts: List[str] = []
        self.actions: List[Dict[str, Any]] = []
        self.observations: List[Any] = []
        self.response: str = ""
        self.confidence: float = 0.0
        self.escalation: bool = False
        self.ticket_id: Optional[str] = None
        self.tokens_used: int = 0
        self.latency_ms: float = 0.0

class SupportAgent:
    """AI-powered customer support agent."""
    
    def __init__(self, vector_db: VectorDB):
        """Initialize the agent."""
        self.vector_db = vector_db
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.conversation_manager = ConversationManager(vector_db=vector_db)
        self.metrics = TokenMetrics(input_tokens=0, output_tokens=0)
        
        # System prompt
        self.system_prompt = """You are an expert customer support agent for a SaaS product.

Your goals:
1. Understand user queries and provide helpful, accurate responses
2. Use available tools to check data, retrieve documents, and create tickets
3. Self-evaluate your confidence in the response
4. Escalate to human support when appropriate

Guidelines:
- Be helpful and professional
- Base answers on retrieved knowledge base documents
- Never make up information not in the KB
- If unsure about something, search the knowledge base first
- If still unsure, escalate to a human agent
- Keep responses concise but complete

Available tools:
- check_order_status: Look up order information
- create_support_ticket: Create a support ticket for complex issues
- search_knowledge_base: Search FAQs and documentation
- escalate_to_human: Escalate to human agent

Response format:
Provide your response in JSON with the following structure:
{
    "thought": "Your reasoning",
    "action": "next action to take",
    "tool_name": "name of tool to use or 'none' if done",
    "tool_input": {},
    "response": "response to user",
    "confidence": 0.0-1.0,
    "escalate": true/false
}"""
        
        logger.info("Initialized SupportAgent")
    
    def retrieve_context(self, query: str, metadata_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant context from vector DB."""
        start_time = time.time()
        
        # Hybrid retrieval with reranking
        documents = self.vector_db.hybrid_retrieve(
            query=query,
            top_k=Config.RETRIEVAL_TOP_K,
            metadata_filter=metadata_filter,
            rerank=True
        )
        
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Retrieved {len(documents)} documents in {latency_ms:.0f}ms")
        
        return documents
    
    def format_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents for inclusion in prompt."""
        if not documents:
            return "No relevant documents found."
        
        formatted = "Retrieved Knowledge Base Documents:\n"
        for i, doc in enumerate(documents, 1):
            formatted += f"\n[{i}] (Score: {doc.get('rerank_score', doc.get('hybrid_score', 0)):.2f})\n"
            formatted += f"{doc['text'][:300]}...\n"
        
        return formatted
    
    def call_llm(self, messages: List[Dict[str, str]]) -> tuple[str, int, int]:
        """Call Claude API and get response."""
        start_time = time.time()
        
        response = self.client.messages.create(
            model=Config.LLM_MODEL,
            max_tokens=1024,
            system=self.system_prompt,
            messages=messages
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract content
        content = response.content[0].text if response.content else ""
        
        # Track token usage
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        
        logger.debug(f"LLM call completed in {latency_ms:.0f}ms | Tokens: {input_tokens} + {output_tokens}")
        
        return content, input_tokens, output_tokens
    
    def parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from LLM."""
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            # Fallback if LLM didn't return JSON
            return {
                "thought": "Parsing error",
                "action": "respond",
                "tool_name": "none",
                "response": response_text,
                "confidence": 0.5,
                "escalate": False
            }
        
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON response")
            return {
                "thought": "Parsing error",
                "action": "respond",
                "tool_name": "none",
                "response": response_text,
                "confidence": 0.5,
                "escalate": False
            }
    
    def execute_action(self, parsed_response: Dict[str, Any]) -> Any:
        """Execute the tool action."""
        tool_name = parsed_response.get("tool_name", "none")
        
        if tool_name == "none":
            return None
        
        try:
            tool_input = parsed_response.get("tool_input", {})
            logger.info(f"Executing tool: {tool_name}")
            result = execute_tool(tool_name, tool_input)
            return result
        
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {"error": str(e)}
    
    def process_query(self, query: str, max_iterations: int = 5) -> Dict[str, Any]:
        """
        Process a user query through the agent loop.
        
        Args:
            query: User query
            max_iterations: Maximum iterations of the agent loop
        
        Returns:
            Agent response with metadata
        """
        logger.info(f"Processing query: {query[:100]}...")
        start_time = time.time()
        
        # Add to conversation memory
        self.conversation_manager.add_user_message(query)
        
        # Retrieve context from KB
        documents = self.retrieve_context(query)
        self.vector_db.hybrid_retrieve(query, top_k=5, rerank=True)
        
        # Format documents for prompt
        formatted_docs = self.format_documents(documents)
        
        # Get conversation context
        conversation_context = self.conversation_manager.get_context()
        
        # Build initial message for LLM
        user_message = f"""User Query: {query}

{formatted_docs}

Conversation Context:
{conversation_context if conversation_context else 'New conversation'}

Please analyze this query and provide your response in JSON format."""
        
        messages = [{"role": "user", "content": user_message}]
        
        # Agent loop
        iteration = 0
        final_response = None
        total_input_tokens = 0
        total_output_tokens = 0
        
        while iteration < max_iterations:
            iteration += 1
            logger.debug(f"Agent iteration {iteration}/{max_iterations}")
            
            # Call LLM
            response_text, input_tokens, output_tokens = self.call_llm(messages)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            
            # Parse response
            parsed = self.parse_llm_response(response_text)
            
            logger.debug(f"LLM Thought: {parsed.get('thought', 'N/A')}")
            logger.debug(f"LLM Tool: {parsed.get('tool_name', 'none')}")
            
            # Check if escalation requested
            if parsed.get("escalate", False) or parsed.get("confidence", 0) < Config.CONFIDENCE_THRESHOLD:
                logger.info("Escalating to human agent")
                final_response = {
                    "response": parsed.get("response", "I need to escalate this to a human agent for better assistance."),
                    "escalated": True,
                    "confidence": parsed.get("confidence", 0),
                    "reasoning": "Confidence below threshold or explicit escalation requested"
                }
                break
            
            # Execute tool if needed
            tool_name = parsed.get("tool_name", "none")
            if tool_name != "none":
                observation = self.execute_action(parsed)
                logger.debug(f"Tool observation: {str(observation)[:200]}")
                
                # Add to conversation
                messages.append({"role": "assistant", "content": response_text})
                
                # Continue conversation with observation
                observation_message = f"""Tool {tool_name} returned:
{json.dumps(observation, indent=2) if observation else "No result"}

Based on this, please provide your response in JSON format."""
                
                messages.append({"role": "user", "content": observation_message})
            else:
                # Done with tools, this is the final response
                final_response = {
                    "response": parsed.get("response", "I'm unable to provide assistance at this time."),
                    "escalated": False,
                    "confidence": parsed.get("confidence", 0.5),
                    "reasoning": parsed.get("thought", "")
                }
                break
        
        # Record metrics
        latency_ms = (time.time() - start_time) * 1000
        
        if not final_response:
            final_response = {
                "response": "Maximum iterations reached. Please contact support.",
                "escalated": True,
                "confidence": 0.2,
                "reasoning": "Max iterations exceeded"
            }
        
        # Add to conversation memory
        self.conversation_manager.add_assistant_message(
            final_response["response"],
            reasoning=final_response.get("reasoning", ""),
            tool_calls=[parsed.get("tool_name", "none")] if parsed.get("tool_name") != "none" else []
        )
        
        return {
            "query": query,
            "response": final_response["response"],
            "escalated": final_response.get("escalated", False),
            "confidence": final_response.get("confidence", 0),
            "documents_used": len(documents),
            "tokens_used": total_input_tokens + total_output_tokens,
            "latency_ms": latency_ms,
            "iterations": iteration
        }
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        messages = self.conversation_manager.short_term.get_messages()
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp
            }
            for msg in messages
        ]
    
    def reset_conversation(self) -> None:
        """Reset the current conversation."""
        self.conversation_manager.reset()
        logger.info("Reset conversation")
