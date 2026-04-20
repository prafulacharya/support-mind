"""Tool definitions for the support agent."""

import json
from typing import Dict, Any, List
from datetime import datetime, timedelta
import random
from pydantic import BaseModel, Field
from utils.logging import logger

# Tool input/output schemas
class CheckOrderInput(BaseModel):
    order_id: str = Field(description="The order ID to check")

class CheckOrderOutput(BaseModel):
    success: bool
    order_id: str
    status: str
    estimated_delivery: str
    tracking_number: str
    items: List[str]

class CreateTicketInput(BaseModel):
    subject: str = Field(description="Support ticket subject")
    description: str = Field(description="Detailed description of the issue")
    priority: str = Field(description="Priority level: low, medium, high, urgent", default="medium")

class CreateTicketOutput(BaseModel):
    success: bool
    ticket_id: str
    status: str
    estimated_resolution_time: str

class SearchKBInput(BaseModel):
    query: str = Field(description="Search query for knowledge base")
    category: str = Field(description="Category to search in (optional)", default=None)

class SearchKBOutput(BaseModel):
    success: bool
    results: List[Dict[str, str]]
    total_found: int

# Mock data
MOCK_ORDERS = {
    "ORD-001": {
        "status": "shipped",
        "items": ["Product A", "Product B"],
        "tracking": "TRACK-123456",
        "delivery_date": (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
    },
    "ORD-002": {
        "status": "processing",
        "items": ["Product C"],
        "tracking": None,
        "delivery_date": None
    },
    "ORD-003": {
        "status": "delivered",
        "items": ["Product A"],
        "tracking": "TRACK-789012",
        "delivery_date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    }
}

MOCK_FAQS = [
    {
        "question": "How do I reset my password?",
        "answer": "Go to login page, click 'Forgot Password', enter your email, and follow the link sent to your inbox."
    },
    {
        "question": "What is your refund policy?",
        "answer": "We offer 30-day money-back guarantee for all products. Contact support with your order ID."
    },
    {
        "question": "How do I track my order?",
        "answer": "Check your confirmation email for a tracking link. You can also log in to your account and view order status."
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept credit cards, debit cards, PayPal, and bank transfers."
    },
    {
        "question": "How long does shipping take?",
        "answer": "Standard shipping: 5-7 business days. Express shipping: 2-3 business days."
    }
]

# Tool implementations
def check_order_status(order_id: str) -> CheckOrderOutput:
    """
    Check the status of an order.
    
    Args:
        order_id: Order ID to check
    
    Returns:
        Order status and details
    """
    logger.info(f"Tool: check_order_status(order_id={order_id})")
    
    if order_id in MOCK_ORDERS:
        order = MOCK_ORDERS[order_id]
        return CheckOrderOutput(
            success=True,
            order_id=order_id,
            status=order["status"],
            estimated_delivery=order.get("delivery_date", "Unknown"),
            tracking_number=order.get("tracking", "Not yet available"),
            items=order["items"]
        )
    else:
        return CheckOrderOutput(
            success=False,
            order_id=order_id,
            status="not_found",
            estimated_delivery="",
            tracking_number="",
            items=[]
        )

def create_support_ticket(subject: str, description: str, priority: str = "medium") -> CreateTicketOutput:
    """
    Create a support ticket.
    
    Args:
        subject: Ticket subject
        description: Detailed description
        priority: Priority level (low, medium, high, urgent)
    
    Returns:
        Ticket creation confirmation
    """
    logger.info(f"Tool: create_support_ticket(subject={subject}, priority={priority})")
    
    # Generate ticket ID
    ticket_id = f"TKT-{random.randint(100000, 999999)}"
    
    # Estimate resolution time based on priority
    resolution_times = {
        "low": "5-7 business days",
        "medium": "2-3 business days",
        "high": "24 hours",
        "urgent": "2 hours"
    }
    
    return CreateTicketOutput(
        success=True,
        ticket_id=ticket_id,
        status="created",
        estimated_resolution_time=resolution_times.get(priority, "3 business days")
    )

def search_knowledge_base(query: str, category: str = None) -> SearchKBOutput:
    """
    Search the knowledge base for FAQs.
    
    Args:
        query: Search query
        category: Optional category filter
    
    Returns:
        Search results
    """
    logger.info(f"Tool: search_knowledge_base(query={query}, category={category})")
    
    query_lower = query.lower()
    results = []
    
    for faq in MOCK_FAQS:
        # Simple keyword matching
        if any(word in faq["question"].lower() for word in query_lower.split()):
            results.append({
                "question": faq["question"],
                "answer": faq["answer"]
            })
    
    return SearchKBOutput(
        success=len(results) > 0,
        results=results,
        total_found=len(results)
    )

def escalate_to_human(ticket_id: str, reason: str) -> Dict[str, Any]:
    """
    Escalate the conversation to a human agent.
    
    Args:
        ticket_id: Associated ticket ID
        reason: Reason for escalation
    
    Returns:
        Escalation confirmation
    """
    logger.warning(f"Tool: escalate_to_human(ticket_id={ticket_id}, reason={reason})")
    
    return {
        "success": True,
        "ticket_id": ticket_id,
        "status": "escalated_to_human",
        "message": f"Your ticket has been escalated. A human agent will contact you within 30 minutes.",
        "queue_position": random.randint(1, 5)
    }

# Tool definitions for LangGraph
TOOLS = [
    {
        "name": "check_order_status",
        "description": "Check the status of an order using the order ID",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID (e.g., ORD-001)"
                }
            },
            "required": ["order_id"]
        },
        "function": check_order_status
    },
    {
        "name": "create_support_ticket",
        "description": "Create a support ticket for complex issues that can't be resolved immediately",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Ticket subject"
                },
                "description": {
                    "type": "string",
                    "description": "Detailed description of the issue"
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "urgent"],
                    "description": "Priority level",
                    "default": "medium"
                }
            },
            "required": ["subject", "description"]
        },
        "function": create_support_ticket
    },
    {
        "name": "search_knowledge_base",
        "description": "Search the knowledge base (FAQs and documentation)",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "category": {
                    "type": "string",
                    "description": "Optional category to filter results"
                }
            },
            "required": ["query"]
        },
        "function": search_knowledge_base
    },
    {
        "name": "escalate_to_human",
        "description": "Escalate the conversation to a human support agent when unable to resolve",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {
                    "type": "string",
                    "description": "The ticket ID for reference"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for escalation"
                }
            },
            "required": ["ticket_id", "reason"]
        },
        "function": escalate_to_human
    }
]

def execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> Any:
    """Execute a tool by name."""
    tool_map = {
        "check_order_status": check_order_status,
        "create_support_ticket": create_support_ticket,
        "search_knowledge_base": search_knowledge_base,
        "escalate_to_human": escalate_to_human
    }
    
    if tool_name not in tool_map:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    func = tool_map[tool_name]
    return func(**tool_input)
