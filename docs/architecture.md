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
                 Embedding
                   Model
                      │
                      ▼
              Vector Search
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
Relevant Books
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
      Preference       Retrieval       Tool
         Agent           Agent         Agent
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

Phase 5

User Interaction
       │
       ▼
Preference Extraction
       │
       ▼
User Memory
       │
       ▼
Future Recommendations


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

Phase 7

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