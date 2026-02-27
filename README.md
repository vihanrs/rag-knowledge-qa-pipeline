# RAG Knowledge QA Pipeline

A multi-agent Retrieval-Augmented Generation (RAG) system built with LangChain, LangGraph, Pinecone, and FastAPI. The pipeline decomposes complex questions into sub-queries, retrieves relevant document chunks from a vector store, and produces verified answers through a chain of specialized agents.

**Frontend Repository:** [vihanrs/rag-knowledge-qa-pipeline-FE](https://github.com/vihanrs/rag-knowledge-qa-pipeline-FE)

---

## Architecture

```
POST /qa
    │
    ▼
[Planning Agent]         Decomposes question → plan + sub-questions
    │
    ▼
[Retrieval Agent]        Calls retrieval_tool once per sub-question
    │                    Collects all ToolMessages → structured context
    ▼
[Summarization Agent]    Generates draft answer from context only
    │
    ▼
[Verification Agent]     Cross-checks draft against context, removes hallucinations
    │
    ▼
QAResponse (answer, context, plan, sub_questions, retrieval_traces)
```

### Agent Pipeline (LangGraph)

```
START → planning → retrieval → summarization → verification → END
```

---

## Project Structure

```
src/app/
├── api.py                          # FastAPI endpoints (/qa, /index-pdf)
├── models.py                       # Pydantic request/response schemas
├── services/
│   ├── qa_service.py               # Facade over LangGraph QA flow
│   └── indexing_service.py         # PDF indexing entry point
└── core/
    ├── config.py                   # Settings (Pydantic Settings, env vars)
    ├── llm/
    │   └── factory.py              # ChatOpenAI factory
    ├── retrieval/
    │   ├── vector_store.py         # Pinecone setup, retrieve(), index_documents()
    │   └── serialization.py        # Document → formatted chunk string
    └── agents/
        ├── state.py                # QAState TypedDict (LangGraph state schema)
        ├── prompts.py              # System prompts for all four agents
        ├── tools.py                # retrieval_tool (Pinecone search)
        ├── agents.py               # Agent definitions + node functions
        └── graph.py                # LangGraph graph wiring + run_qa_flow()
```

---

## LangGraph State

`QAState` carries all data through the pipeline:

| Field | Type | Set by |
|---|---|---|
| `question` | `str` | Initial input |
| `plan` | `str \| None` | Planning node |
| `sub_questions` | `list[str] \| None` | Planning node |
| `retrieval_traces` | `str \| None` | Retrieval node |
| `raw_context_blocks` | `list[str] \| None` | Retrieval node |
| `context` | `str \| None` | Retrieval node |
| `draft_answer` | `str \| None` | Summarization node |
| `answer` | `str \| None` | Verification node |

---

## Features

### Feature 1: Query Planning & Decomposition

Complex questions are analyzed by a **Planning Agent** before any retrieval occurs.

- Decomposes the question into 2–4 focused sub-questions
- Produces a plain-English search plan
- Sub-questions are passed to the Retrieval Agent to guide targeted tool calls

**Example:**
```
Question: "What are the advantages of vector databases compared to traditional
databases, and how do they handle scalability?"

Plan: Search for vector DB advantages, then compare with relational DBs, then
cover scalability mechanisms.

Sub-questions:
- vector database advantages and benefits
- vector database vs relational database comparison
- vector database scalability architecture
```

### Feature 2: Multi-Call Retrieval with Message Organization

The Retrieval Agent makes **one tool call per sub-question**. All tool call results are captured and organized — nothing is silently discarded.

- Every `ToolMessage` in the agent's message history is collected
- The query used for each call is extracted from the preceding `AIMessage.tool_calls`
- Context passed downstream is structured per retrieval call
- A human-readable `retrieval_traces` log is produced

**Structured context format:**
```
=== RETRIEVAL CALL 1 (query: "vector database advantages") ===

Chunk 1 (page=3): ...
Chunk 2 (page=7): ...

=== RETRIEVAL CALL 2 (query: "vector database vs relational") ===

Chunk 1 (page=5): ...
```

**Retrieval trace format:**
```
Retrieval Call 1:
Query: "vector database advantages"
Chunks Retrieved: 4
Sources: Pages 3, 7, 12, 15

Retrieval Call 2:
Query: "vector database vs relational"
Chunks Retrieved: 4
Sources: Pages 5, 8, 9, 11
```

---

## API

### `POST /qa`

Submit a question and receive a verified answer with full pipeline transparency.

**Request:**
```json
{ "question": "What are the advantages of vector databases?" }
```

**Response:**
```json
{
  "answer": "Vector databases offer...",
  "context": "=== RETRIEVAL CALL 1 (query: \"...\") ===\n\nChunk 1 (page=3): ...",
  "plan": "Search for vector DB advantages then compare with relational DBs.",
  "sub_questions": [
    "vector database advantages benefits",
    "vector database vs relational database"
  ],
  "retrieval_traces": "Retrieval Call 1:\nQuery: \"...\"\nChunks Retrieved: 4\nSources: Pages 3, 7"
}
```

### `POST /index-pdf`

Upload a PDF and index it into the Pinecone vector store.

**Request:** Multipart form with a `file` field (PDF only).

**Response:**
```json
{
  "filename": "document.pdf",
  "chunks_indexed": 42,
  "message": "PDF indexed successfully."
}
```

---

## Setup

### Prerequisites

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv) (dependency manager)
- OpenAI API key
- Pinecone API key + index

### Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL_NAME=gpt-4o-mini
OPENAI_EMBEDDING_MODEL_NAME=text-embedding-3-large

PINECONE_API_KEY=...
PINECONE_INDEX_NAME=...

RETRIEVAL_K=4
```

### Install & Run

```bash
uv sync
uv run uvicorn src.app.api:app --reload
```

API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## Tech Stack

| Component | Library |
|---|---|
| Agent orchestration | `langgraph` |
| LLM + agents | `langchain`, `langchain-openai` |
| Vector store | `pinecone`, `langchain-pinecone` |
| PDF loading | `pypdf` |
| Text splitting | `langchain-text-splitters` |
| API | `fastapi`, `uvicorn` |
| Config | `pydantic-settings` |
