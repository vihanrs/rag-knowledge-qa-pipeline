from typing import Any

from pydantic import BaseModel


class VercelChatRequest(BaseModel):
    """Request body sent by Vercel AI SDK useChat hook to /qa/stream.

    Uses raw dicts for messages so the schema accepts both the legacy
    {role, content: str} format and the AI SDK v5 UI message format
    {id, role, parts: [{type, text}]}.
    """

    id: str | None = None
    messages: list[dict[str, Any]]


class QuestionRequest(BaseModel):
    """Request body for the `/qa` endpoint.

    The PRD specifies a single field named `question` that contains
    the user's natural language question about the vector databases paper.
    """

    question: str


class QAResponse(BaseModel):
    """Response body for the `/qa` endpoint.

    From the API consumer's perspective we expose the final verified answer,
    the retrieved context, and the planning output (plan + sub_questions)
    so consumers can inspect the query decomposition strategy.
    """

    answer: str
    context: str
    plan: str | None = None
    sub_questions: list[str] | None = None
    retrieval_traces: str | None = None
