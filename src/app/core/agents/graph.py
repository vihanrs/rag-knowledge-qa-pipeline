"""LangGraph orchestration for the linear multi-agent QA flow."""

import uuid
from functools import lru_cache
from typing import Any, AsyncIterator, Dict

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from .agents import planning_node, retrieval_node, summarization_node, verification_node
from .state import QAState


def create_qa_graph() -> Any:
    """Create and compile the linear multi-agent QA graph.

    The graph executes in order:
    1. Planning Agent: decomposes the question into a plan + sub-questions
    2. Retrieval Agent: gathers context from vector store using the plan
    3. Summarization Agent: generates draft answer from context
    4. Verification Agent: verifies and corrects the answer

    Returns:
        Compiled graph ready for execution.
    """
    builder = StateGraph(QAState)

    # Add nodes for each agent
    builder.add_node("planning", planning_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("summarization", summarization_node)
    builder.add_node("verification", verification_node)

    # Define linear flow: START -> planning -> retrieval -> summarization -> verification -> END
    builder.add_edge(START, "planning")
    builder.add_edge("planning", "retrieval")
    builder.add_edge("retrieval", "summarization")
    builder.add_edge("summarization", "verification")
    builder.add_edge("verification", END)

    return builder.compile()


@lru_cache(maxsize=1)
def get_qa_graph() -> Any:
    """Get the compiled QA graph instance (singleton via LRU cache)."""
    return create_qa_graph()


def run_qa_flow(question: str) -> Dict[str, Any]:
    """Run the complete multi-agent QA flow for a question.

    This is the main entry point for the QA system. It:
    1. Initializes the graph state with the question
    2. Executes the linear agent flow (Retrieval -> Summarization -> Verification)
    3. Extracts and returns the final results

    Args:
        question: The user's question about the vector databases paper.

    Returns:
        Dictionary with keys:
        - `answer`: Final verified answer
        - `draft_answer`: Initial draft answer from summarization agent
        - `context`: Retrieved context from vector store
    """
    graph = get_qa_graph()

    initial_state: QAState = {
        "question": question,
        "plan": None,
        "sub_questions": None,
        "retrieval_traces": None,
        "raw_context_blocks": None,
        "context": None,
        "draft_answer": None,
        "answer": None,
    }

    final_state = graph.invoke(initial_state)

    return final_state


def _sse(data: dict) -> str:
    """Format a dict as a Server-Sent Event string."""
    import json
    return f"data: {json.dumps(data)}\n\n"


async def run_qa_stream(question: str) -> AsyncIterator[str]:
    """Async streaming entry point for Vercel AI SDK.

    Streams Vercel Data Stream Protocol v1 events:
    - tool-input-available + tool-output-available for planning, retrieval, summarization
    - start/text-start/text-delta/text-end/finish for the final verified answer

    Args:
        question: The user's question about the vector databases paper.

    Yields:
        SSE-formatted strings for the Vercel AI SDK frontend.
    """
    import json

    graph = get_qa_graph()

    initial_state: QAState = {
        "question": question,
        "plan": None,
        "sub_questions": None,
        "retrieval_traces": None,
        "raw_context_blocks": None,
        "context": None,
        "draft_answer": None,
        "answer": None,
        "messages": [],
    }
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Track which fields have already been streamed to avoid re-emitting
    # on subsequent chunks (stream_mode="values" emits full state each time).
    emitted: set[str] = set()

    # Map state field -> (tool_name, callable that returns the display string)
    tool_steps = [
        ("plan", "planning", lambda s: (
            s["plan"] + (
                "\n\n" + "\n".join(f"• {q}" for q in s["sub_questions"])
                if s.get("sub_questions") else ""
            )
        )),
        ("retrieval_traces", "retrieval", lambda s: s["retrieval_traces"]),
        ("draft_answer", "summarization", lambda s: s["draft_answer"]),
    ]

    CHUNK_SIZE = 50

    try:
        async for chunk in graph.astream(initial_state, config, stream_mode="values"):
            # --- Emit tool pairs for intermediate steps (once per field) ---
            for field, tool_name, get_output in tool_steps:
                if field in emitted:
                    continue
                value = chunk.get(field)
                if not value:
                    continue

                emitted.add(field)
                tool_call_id = str(uuid.uuid4())
                output_text = get_output(chunk)

                yield _sse({
                    "type": "tool-input-available",
                    "toolCallId": tool_call_id,
                    "toolName": tool_name,
                    "input": {"question": question},
                })
                yield _sse({
                    "type": "tool-output-available",
                    "toolCallId": tool_call_id,
                    "output": output_text,
                })

            # --- Stream final answer from verification node ---
            messages = chunk.get("messages") or []
            if messages and "answer" not in emitted:
                from langchain_core.messages import AIMessage as AI
                last = messages[-1]
                if isinstance(last, AI) and last.content and last.content.strip():
                    emitted.add("answer")
                    msg_id = f"msg_{uuid.uuid4().hex[:12]}"

                    yield _sse({"type": "start", "messageId": msg_id})
                    yield _sse({"type": "text-start", "id": msg_id})

                    content = last.content
                    for i in range(0, len(content), CHUNK_SIZE):
                        yield _sse({
                            "type": "text-delta",
                            "id": msg_id,
                            "delta": content[i:i + CHUNK_SIZE],
                        })

                    yield _sse({"type": "text-end", "id": msg_id})
                    yield _sse({"type": "finish-step", "finishReason": "stop"})

        yield _sse({"type": "finish", "finishReason": "stop"})
        yield "data: [DONE]\n\n"

    except Exception as e:
        yield _sse({"type": "error", "errorText": str(e)})
