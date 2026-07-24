# BookBuddy Requirements

## Vision

BookBuddy is an AI-powered book discovery platform that helps readers find books through explainable recommendations instead of traditional similarity-based suggestions.

---

## Problem Statement

Most recommendation systems answer:

> "People who liked this also liked..."

BookBuddy aims to answer:

> "Based on what you enjoy reading, here's why this book fits your preferences."

The focus is on transparency, personalization, and conversational discovery.

---

## Target Users

- Readers looking for their next book
- Readers who want recommendations beyond bestseller lists
- Users who prefer natural language over filters and categories

---

## MVP Goals

- Import and index the Goodreads dataset
- Search books
- View book details
- Perform semantic book search
- Ask for recommendations in natural language
- Explain recommendations using an LLM

---

## Non Goals

The first version will not include:

- User authentication
- Reading progress tracking
- Social features
- Book reviews
- Reading challenges
- Book purchasing

---

## Functional Requirements

### Dataset

- Load Goodreads dataset
- Clean and normalize records
- Store searchable book metadata

### Search

- Search by title
- Search by author
- Semantic search using embeddings

### Recommendations

- Recommend books from natural language queries
- Explain why a recommendation was made

Example:

> Recommend books similar to *Project Hail Mary*, but with less science and stronger character development.

---

## Non Functional Requirements

- Support millions of books
- Fast semantic retrieval
- Modular AI architecture
- Explainable recommendations
- Extensible pipeline for future AI capabilities

---

## AI Scope (Initial)

The MVP will explore:

- LLM-powered recommendation explanations
- Retrieval-Augmented Generation (RAG)
- Vector search
- User preference extraction

Future iterations may include:

- Multi-agent workflows
- MCP integrations
- Long-term memory
- Personalized user profiles

---

## Success Criteria

A user should be able to describe the type of book they are looking for in natural language and receive recommendations together with a clear explanation of why each book was selected.