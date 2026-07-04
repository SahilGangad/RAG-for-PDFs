# Evaluation: Local RAG Document Q&A

## Setup

- **Corpus:** Single-topic PDF (statistics reference document, "Simple Linear Regression")
- **Embedding model:** `nomic-embed-text` (via Ollama), 768-dim, retrieval-tuned, requires `search_document:` / `search_query:` prefixes
- **Vector store:** ChromaDB, persistent, L2 distance
- **Generation models tested:** Llama 3 8B (local, via Ollama), `openai/gpt-oss-20b` (hosted, via Groq)
- **Chunking:** 1000 chars, 200-char overlap
- **Retrieval:** top-k = 3

## Method

Five query categories were used to stress-test different parts of the pipeline:

| Category | Example |
|---|---|
| Direct fact | "What do β₀ and β₁ represent?" |
| Paraphrase | "Why take the log of a predictor if a scatterplot looks curved?" |
| Specific number | "What R-squared value did the model achieve?" |
| Multi-hop | "Why was ln(Discharge) used, and what was the correlation coefficient?" |
| Out-of-scope | "What does this document say about multiple regression with 3+ predictors?" |

Each category was run multiple times, and retrieved chunks were manually cross-checked against generated answers rather than trusting the answer at face value.

## Findings

### 1. Retrieval handles single-topic queries well, including true paraphrase
Direct-fact and paraphrased queries reliably surfaced the correct chunk in the top-3, even when the query shared no exact vocabulary with the source text (e.g. asking about "taking the log of a variable" correctly retrieved the "Tukey's bulging rule" section). This confirms retrieval is doing semantic matching, not keyword overlap — validated by deliberately choosing paraphrases with zero lexical overlap.

### 2. Naive top-k retrieval fails silently on multi-hop questions
A single query embedding represents one point in vector space. A two-part question (e.g. "why X, and what was Y") pulls toward whichever half has the stronger semantic signal, and can drop the other half entirely — with no signal to the user that anything is missing. In testing, the correlation-coefficient half was consistently retrieved; the "why" half was consistently missed, because the two answers live in different, distant sections of the document.

**Attempted fix:** query decomposition (splitting the question into independent sub-questions via an LLM call, then retrieving separately for each) successfully diversified the retrieved chunks, but did not fully resolve this case — the decomposed sub-question, when anchored to specific example details ("in the TDS example"), still failed to retrieve the general-principle chunk it needed. This suggests decomposition helps but isn't sufficient on its own; combining it with a higher top-k or a relevance-filtering step would likely be needed for full resolution.

### 3. Locally-quantized Llama 3 shows non-determinism even at temperature=0 with a fixed seed
The same query, same retrieved context, same `temperature=0`/`seed=42` settings produced differently-worded (and differently-reasoned) answers across repeated runs. This is a known characteristic of quantized local inference (floating-point non-associativity under batching) rather than a bug in the calling code. Implication: a single test run is not sufficient to validate a local-LLM RAG pipeline's behavior — multiple runs per query are needed to distinguish a stable result from noise.

### 4. Larger/more capable models can silently violate grounding instructions
When the retrieved context did not contain the answer to the "why" half of the multi-hop question, `openai/gpt-oss-20b` produced a fluent, factually correct explanation (log transformation linearizes a curvilinear relationship) — but this answer was **not present in the retrieved chunks**. The model filled the gap from its own pretraining knowledge, despite an explicit system-prompt instruction to answer "strictly based on the context" and to say "I don't know" if the answer isn't covered.

This is distinct from hallucination in the usual sense (inventing something false): the content was true, but the grounding constraint was violated in a way that is **undetectable without manually cross-referencing the answer against the retrieved source chunks** — the answer reads as if it came from the document. This is arguably a more concerning failure mode for a RAG system than an obviously wrong answer, since it defeats the main trust benefit RAG is meant to provide (verifiable grounding).

### 5. The system prompt reliably prevents both under- and over-confidence in clear cases
When retrieval returned only tangentially related chunks for a genuinely out-of-scope question (multiple regression with 3+ predictors), both models correctly responded "I don't know based on the provided document" rather than stretching the tangential content into an answer. The grounding instruction works precisely in the cases where relevance is clearly present or clearly absent — it's the ambiguous middle ground (partially-relevant, topically-adjacent chunks) where it breaks down.

## Summary

| Scenario | Retrieval | Generation |
|---|---|---|
| Direct fact | Reliable | Correct |
| True paraphrase | Reliable | Correct |
| Specific figure | Reliable | Correct |
| Multi-hop | Incomplete (single-vector limit) | Fills gap via fabrication or pretraining knowledge, not always flagged |
| Out-of-scope | Correctly low-confidence | Correctly refuses |

**Core conclusion:** naive top-k RAG is reliable for single-fact and single-topic semantic queries, but has two compounding failure modes on multi-part questions — a retrieval-side gap (single embedding can't cover two topics) and a generation-side gap (models may paper over missing context using fluent, ungrounded, sometimes-correct answers rather than admitting the gap). Query decomposition partially addresses the first; the second remains an open problem best mitigated by explicit source-attribution checks (e.g., requiring the model to cite which chunk supports each claim) — noted here as future work.

## Future work
- Query decomposition + relevance filtering (score each chunk against each sub-question before generation)
- Per-claim source attribution in the generated answer, to make grounding violations detectable automatically rather than requiring manual chunk inspection
- Multi-query retrieval (retrieve for the original question AND its decomposed sub-questions, merge and re-rank)
- Increase top-k specifically for questions detected as multi-part
