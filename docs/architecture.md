# BookBuddy Architecture

## 1. Overview

BookBuddy is an AI-powered book discovery platform built on the Goodreads Books dataset.

The system will evolve incrementally from a conventional full-stack application into an AI-native recommendation system.

The architecture is divided into phases so that each AI capability is introduced when there is a clear technical purpose for it.

---

# 2. Architecture Evolution

```text
Phase 1
Data Ingestion + Book Search
        ↓
Phase 2
Semantic Search + Embeddings
        ↓
Phase 3
RAG + LLM Recommendations
        ↓
Phase 4
Multi-Agent Recommendation System
        ↓
Phase 5
Persistent User Memory
        ↓
Phase 6
MCP-based External Tools
        ↓
Phase 7
Fine-Tuning + Quantized Local Inference


Phase 1

┌──────────────────────┐
│ Goodreads Dataset    │
│ Raw CSV              │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Ingestion Pipeline   │
│ Python / Polars      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ PostgreSQL            │
│ Book Metadata         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ FastAPI               │
│ Application API       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Next.js / React       │
│ Web Application       │
└──────────────────────┘

Phase 2

                    User
                      │
                      ▼
                 Next.js
                      │
                      ▼
                  FastAPI
                      │
                      ▼
               Embedding Model
                      │
                      ▼
              Vector Store
              (PostgreSQL + pgvector)
                      │
                      ▼
            Semantic Retrieval
                      │
                      ▼
             Relevant Books


Phase 3

User Query
    │
    ▼
Retriever
    │
    ▼
Relevant Book Summaries + Metadata
    │
    ▼
LLM
    │
    ▼
Recommendation + Explanation

Phase 4

                         User
                           │
                           ▼
                  ┌─────────────────┐
                  │ Planner Agent   │
                  └────────┬────────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
      Preference       Retrieval       Internal Tool
         Agent           Agent             Agent
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                 Recommendation Agent
                           │
                           ▼
                    Critic Agent
                           │
                           ▼
                    Final Response

The Internal Tool Agent uses BookBuddy's own application capabilities. External tool access through MCP is introduced in Phase 6.

Phase 5

User Interaction
       │
       ▼
Preference Extraction
       │
       ▼
Long-Term Preference Memory
       │
       ▼
Future Recommendations

Memory represents persistent reading preferences and useful user context rather than simply storing complete conversation transcripts.


Phase 6 

                     Agent
                       │
                       ▼
                 MCP Interface
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      External      External     External
       Tool A        Tool B       Tool C

# 9. Phase 7 — Model Engineering Experiments

Goal

Explore model customization and local inference as separate AI engineering concerns.

Fine-Tuning

Fine-tuning will be evaluated against prompting and retrieval-based approaches rather than being assumed to be necessary.

The objective is to demonstrate the trade-off between:

* Prompting
* RAG
* Fine-tuning

Quantization

Quantization will be explored when running a local model.

The objective is to understand the trade-offs between:

* Model size
* Memory requirements
* Inference speed
* Quality



## AI Engineering Roadmap

BookBuddy is intentionally being developed incrementally from a data/search application into an evaluated AI application.

The goal is not to add technologies for résumé keywords. Each capability should solve a real problem in BookBuddy and provide an implementation that can be explained in an interview.

### Phase 1 — Data and Search Foundation

- Ingest Goodreads dataset
- Profile, clean, and normalize source data
- Store structured data in PostgreSQL
- Build semantic search using embeddings and pgvector
- Support metadata filtering
- Expose search functionality through an API

### Phase 2 — RAG

Use the search foundation to provide grounded recommendations and answers.

- Retrieve relevant books and authors
- Construct context for the LLM
- Generate recommendations/answers grounded in retrieved data
- Include source books/citations
- Handle insufficient retrieval without fabricating answers

### Phase 3 — Agentic Layer

Introduce LangGraph only when multiple tools and decision-making provide a real benefit.

Potential BookBuddy tools:

- Semantic search
- Metadata filtering
- Author lookup
- Recommendations

Concepts to demonstrate:

- Tool calling
- State
- Routing / conditional edges
- Retries
- Fallbacks

The initial design should remain a single agent. Multi-agent architecture should only be introduced if a concrete use case justifies it.

### Phase 4 — MCP

Expose selected BookBuddy capabilities through MCP.

The implementation should demonstrate understanding of:

- MCP server
- MCP client
- Tools
- Tool schemas
- Calling BookBuddy capabilities through MCP

MCP should be added because it provides a useful integration boundary, not simply because it is a current AI technology.

### Phase 5 — Evaluation and Observability

Create a small evaluation dataset of representative BookBuddy queries.

Initially target approximately 20–30 queries.

Evaluate:

- Retrieval quality
- Answer quality
- Grounding
- Regression across changes

Add observability for:

- Model calls
- Latency
- Token usage
- Retrieval behavior
- Failures

A tracing/evaluation platform such as Langfuse may be introduced here.

### Phase 6 — Production Engineering

Add production concerns incrementally:

- Docker
- CI/CD
- Unit tests
- Integration tests
- Input/output validation
- AI security considerations
- Prompt injection awareness
- Tool boundaries
- Authorization where relevant
- Deployment

## Guiding Principle

BookBuddy should evolve through the following progression:

```text
Data project
    ↓
Semantic search
    ↓
RAG
    ↓
Agent
    ↓
MCP
    ↓
Evaluated and observable AI application