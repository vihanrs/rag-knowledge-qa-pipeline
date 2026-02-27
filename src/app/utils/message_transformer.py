"""
Message transformation utilities for converting Vercel AI SDK UI messages
to internal backend format.

Similar to how the AI SDK handles UI message → Model message conversion.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class MessagePart(BaseModel):
    """Individual part of a UI message (text, tool-call, etc.)"""
    type: str
    text: Optional[str] = None


class UIMessage(BaseModel):
    """Vercel AI SDK UI message format"""
    id: str
    role: str
    parts: Optional[List[MessagePart]] = None
    content: Optional[str] = None


def extract_user_message(messages: List[Dict[str, Any]]) -> str:
    """
    Extract the latest user message text from UI messages array.

    The Vercel AI SDK sends messages in this format:
    {
        "id": "msg-id",
        "role": "user",
        "parts": [{"type": "text", "text": "Hello"}],
        "content": "Hello"  # Sometimes present as fallback
    }

    Args:
        messages: List of UI messages from the frontend

    Returns:
        The extracted text content from the latest user message
    """
    if not messages:
        return ""

    # Get the last message (most recent)
    last_message = messages[-1]

    # Try to extract from parts first (preferred)
    if "parts" in last_message and last_message["parts"]:
        for part in last_message["parts"]:
            if part.get("type") == "text" and part.get("text"):
                return part["text"]

    # Fallback to content field if parts not available
    if "content" in last_message and last_message["content"]:
        return last_message["content"]

    return ""


def validate_ui_message_format(data: Dict[str, Any]) -> bool:
    """
    Validate that the request matches Vercel AI SDK format.

    Expected format:
    {
        "id": "conversation-id",
        "messages": [...],
        "trigger": "submit-message"
    }

    Args:
        data: Request body dictionary

    Returns:
        True if format is valid, False otherwise
    """
    required_fields = ["id", "messages", "trigger"]

    for field in required_fields:
        if field not in data:
            return False

    if not isinstance(data["messages"], list):
        return False

    return True
