"""Agent implementations for the multi-agent RAG flow.

This module defines three LangChain agents (Retrieval, Summarization,
Verification) and thin node functions that LangGraph uses to invoke them.
"""

from typing import List

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..llm.factory import create_chat_model
from .prompts import (
    PLANNING_SYSTEM_PROMPT,
    RETRIEVAL_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
)
from .state import QAState
from .tools import retrieval_tool


def _extract_last_ai_content(messages: List[object]) -> str:
    """Extract the content of the last AIMessage in a messages list."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return ""


# Define agents at module level for reuse
planning_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=PLANNING_SYSTEM_PROMPT,
)

retrieval_agent = create_agent(
    model=create_chat_model(),
    tools=[retrieval_tool],
    system_prompt=RETRIEVAL_SYSTEM_PROMPT,
)

summarization_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
)

verification_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=VERIFICATION_SYSTEM_PROMPT,
)


def _parse_plan_response(content: str) -> tuple[str, list[str]]:
    """Parse the planning agent output into (plan, sub_questions)."""
    plan = ""
    sub_questions: list[str] = []

    lines = content.splitlines()
    section = None
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("PLAN:"):
            section = "plan"
            inline = stripped[5:].strip()
            if inline:
                plan = inline
        elif stripped.upper().startswith("SUB_QUESTIONS:"):
            section = "sub_questions"
        elif section == "plan" and stripped and not stripped.startswith("-"):
            plan = (plan + " " + stripped).strip()
        elif section == "sub_questions" and stripped.startswith("-"):
            sq = stripped.lstrip("-").strip()
            if sq:
                sub_questions.append(sq)

    return plan, sub_questions


def planning_node(state: QAState) -> dict:
    """Planning Agent node: decomposes the question into a search strategy.

    This node:
    - Sends the user's question to the Planning Agent.
    - Parses the structured PLAN and SUB_QUESTIONS from the response.
    - Stores them in `state["plan"]` and `state["sub_questions"]`.
    """
    question = state["question"]

    result = planning_agent.invoke({"messages": [HumanMessage(content=question)]})
    messages = result.get("messages", [])
    content = _extract_last_ai_content(messages)

    plan, sub_questions = _parse_plan_response(content)

    print(f"[Planning] Plan: {plan}")
    print(f"[Planning] Sub-questions: {sub_questions}")

    return {
        "plan": plan,
        "sub_questions": sub_questions,
    }


def retrieval_node(state: QAState) -> QAState:
    """Retrieval Agent node: gathers context from vector store.

    This node:
    - Sends the user's question together with the plan and sub-questions
      produced by the Planning Agent to the Retrieval Agent.
    - The agent uses the attached retrieval tool to fetch document chunks,
      calling it once per sub-question for broader coverage.
    - The agent consolidates all retrieved chunks into a final CONTEXT block.
    - Stores the consolidated context string in `state["context"]`.
    """
    question = state["question"]
    plan = state.get("plan") or ""
    sub_questions = state.get("sub_questions") or []

    # Build a rich message that includes the plan and sub-questions so the
    # retrieval agent can make targeted, per-sub-question tool calls.
    if plan or sub_questions:
        sub_q_block = "\n".join(f"  - {sq}" for sq in sub_questions)
        message_content = (
            f"Original Question: {question}\n\n"
            f"Search Plan: {plan}\n\n"
            f"Sub-questions to retrieve for:\n{sub_q_block}"
        )
    else:
        message_content = question

    result = retrieval_agent.invoke({"messages": [HumanMessage(content=message_content)]})

    messages = result.get("messages", [])

    # Pair each ToolMessage with the query from the AIMessage that called it.
    # Message order: ... AIMessage(tool_calls=[{args: {query: "..."}}]), ToolMessage, ...
    raw_context_blocks: list[str] = []
    trace_lines: list[str] = []
    structured_blocks: list[str] = []

    for i, msg in enumerate(messages):
        if not isinstance(msg, ToolMessage):
            continue

        # Extract the query from the preceding AIMessage's tool_calls
        query = "unknown"
        if i > 0:
            prev = messages[i - 1]
            if isinstance(prev, AIMessage) and prev.tool_calls:
                query = prev.tool_calls[0].get("args", {}).get("query", "unknown")

        call_number = len(raw_context_blocks) + 1
        content = str(msg.content)

        # Count chunks and extract page numbers for the trace log
        chunk_lines = [l for l in content.splitlines() if l.startswith("Chunk ")]
        pages = [
            l.split("page=")[1].split(")")[0]
            for l in chunk_lines
            if "page=" in l
        ]
        pages_str = ", ".join(pages) if pages else "unknown"

        raw_context_blocks.append(content)

        trace_lines.append(
            f"Retrieval Call {call_number}:\n"
            f"Query: \"{query}\"\n"
            f"Chunks Retrieved: {len(chunk_lines)}\n"
            f"Sources: Pages {pages_str}"
        )

        structured_blocks.append(
            f'=== RETRIEVAL CALL {call_number} (query: "{query}") ===\n\n{content}'
        )

        print(f"[Retrieval] Call {call_number} | query='{query}' | chunks={len(chunk_lines)}")

    context = "\n\n".join(structured_blocks)
    retrieval_traces = "\n\n".join(trace_lines)

    return {
        "context": context,
        "raw_context_blocks": raw_context_blocks,
        "retrieval_traces": retrieval_traces,
    }


def summarization_node(state: QAState) -> QAState:
    """Summarization Agent node: generates draft answer from context.

    This node:
    - Sends question + context to the Summarization Agent.
    - Agent responds with a draft answer grounded only in the context.
    - Stores the draft answer in `state["draft_answer"]`.
    """
    question = state["question"]
    context = state.get("context")

    user_content = f"Question: {question}\n\nContext:\n{context}"

    result = summarization_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )
    messages = result.get("messages", [])
    draft_answer = _extract_last_ai_content(messages)

    return {
        "draft_answer": draft_answer,
    }


def verification_node(state: QAState) -> QAState:
    """Verification Agent node: verifies and corrects the draft answer.

    This node:
    - Sends question + context + draft_answer to the Verification Agent.
    - Agent checks for hallucinations and unsupported claims.
    - Stores the final verified answer in `state["answer"]`.
    """
    question = state["question"]
    context = state.get("context", "")
    draft_answer = state.get("draft_answer", "")

    user_content = f"""Question: {question}

Context:
{context}

Draft Answer:
{draft_answer}

Please verify and correct the draft answer, removing any unsupported claims."""

    result = verification_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )
    messages = result.get("messages", [])
    answer = _extract_last_ai_content(messages)

    return {
        "answer": answer,
    }
