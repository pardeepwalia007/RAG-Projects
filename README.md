## 🧠 Agentic RAG LangGraph Architecture

![Agentic RAG LangGraph Flow](agentic_rag_langgraph.png)

This diagram illustrates the **agentic control flow** of the RAG system built using **LangGraph**.

The pipeline begins at `__start__`, where the user query enters the system.  
Relevant documents are first retrieved via `retrieve_docs`, combining vector search over embedded CSVs and documents.

The `decide_sql` node acts as an **intent router**, determining whether the query requires:
- purely semantic retrieval, or  
- structured computation using SQL.

If computation is required, execution moves to `run_sql_path`, where SQL is dynamically generated, sanitized, and executed against DuckDB.  
Quantitative results can optionally flow into `visualize_data` for chart generation.

A `reflection_node` enables self-correction and retry logic when SQL execution or reasoning fails, allowing the system to loop back intelligently instead of terminating early.

Finally, all successful paths converge at `summarize`, where retrieved context, computed results, and visual insights are synthesized into a grounded natural-language response before reaching `__end__`.

This architecture demonstrates **agentic decision-making, conditional execution, self-healing logic, and hybrid RAG + SQL reasoning** in a single unified workflow.


## Project Strcuture and Core Components
```text
Code/
├── api.py
├── app_langgraph.py
├── ingestion.py
├── intent_llm.py
├── llm_sql_agent.py
├── pdf_to_markdown.py
├── sql_engine.py
├── sql_orchestrator.py
├── summarization_agent.py
├── tests_logger.py
├── ui.py
├── vectorize.py
```


## ⚙️ Core Application Modules (`Code/`)

### `app_langgraph.py` — Agentic Control Plane
Defines the **LangGraph state machine** that orchestrates the entire RAG workflow.  
Handles conditional routing, retries, and convergence into final response generation.

> Think of this as the **brain** of the system.

---

### `ingestion.py` — Data & Document Ingestion
Responsible for loading CSVs, validating schemas, and preparing raw inputs for downstream processing.

---

### `vectorize.py` — Embedding & Retrieval Layer
- Chunks documents  
- Generates embeddings  
- Builds and queries the vector database  

Used during semantic retrieval phases of RAG.

---

### `intent_llm.py` — Query Intent Classification
Determines **user intent** (semantic lookup vs. analytical computation).  
This drives the routing logic inside LangGraph.

---

### `sql_orchestrator.py` — SQL Decision Logic
Decides **if SQL should run** and enforces safety rules before execution.

---

### `llm_sql_agent.py` — SQL Generation & Execution
- Generates SQL via LLM  
- Sanitizes queries  
- Executes against DuckDB  
- Returns structured results  

This enables **LLM-driven analytics**, not just retrieval.

---

### `sql_engine.py` — DuckDB Execution Layer
Low-level SQL execution utilities and dataframe handling.

---

### `pdf_to_markdown.py` — Document Normalization
Converts PDFs into clean Markdown for embedding and retrieval.

---

### `summarization_agent.py` — Final Answer Synthesis
Combines:
- Retrieved documents  
- SQL results  
- Optional visual insights  

into a grounded natural-language response.

---

### `tests_logger.py` — Observability & Debugging
Captures intermediate outputs, retries, and failures for inspection and evaluation.

---

### `ui.py` — User Interface Layer
Handles user interaction (CLI / lightweight UI), acting as the system entry point.

---

## Visual Representations 

### 1) End-to-end request flow (UI → API → LangGraph)

```mermaid
  
flowchart LR
  UI["Streamlit UI - Code/ui.py"] -->|"POST /query"| API["FastAPI - Code/api.py"]
  API -->|"graph.invoke(state)"| LG["LangGraph Orchestrator - Code/app_langgraph.py"]
  LG --> API
  API --> UI

```
### 2) LangGraph core routing
```mermaid

flowchart TD
  S["START"] --> RD["retrieve_docs"]
  RD --> DS["decide_sql"]

  DS -->|"run_sql_path"| SQL["run_sql_path"]
  DS -->|"summarize"| SUM["summarize"]
  DS -->|"retrieve-only"| RET["retrieve_docs"]

  SQL -->|"success"| VIZ["visualize_data"]
  SQL -->|"needs retry"| REF["reflection_node"]

  REF --> DS
  VIZ --> SUM
  SUM --> E["END"]
```
### 3) Runtime build (CSV + PDFs → DuckDB + Retriever)
``` mermaid
flowchart TD
  CSV["CSV files"] --> DDB["DuckDB runtime - sql_engine.py"]
  PDF["PDF files"] --> P2M["pdf_to_markdown.py"]
  P2M --> CH["Markdown chunks"]
  CH --> RET["Retriever build - vectorize.py"]
  RET --> VDB["Vector DB"]
  DDB --> GRAPH["LangGraph run - app_langgraph.py"]
  VDB --> GRAPH

```

### 4) Evaluation pipeline (questions → runs → RAGAS report)

``` mermaid
flowchart LR
  Q["eval_questions.jsonl"] --> RUN["run_eval.py"]
  RUN --> OUT["run_outputs - raw results"]
  OUT --> SCORE["score_ragas.py"]
  SCORE --> REP["reports - RAGAS metrics"]
```

## Cluster-level mini Mermaids

### 5) Retrieval subsystem

``` mermaid
flowchart TD
  PDF["PDF inputs"] --> P2M["pdf_to_markdown.py"]
  P2M --> MD["Markdown"]
  MD --> VEC["vectorize.py - chunk and embed"]
  VEC --> VDB["Vector DB"]

  Q["User question"] --> RET["retrieve_docs"]
  VDB --> RET
  RET --> DOCS["Top context chunks"]
```

### 6) Intent + routing subsystem
``` mermaid
flowchart LR
  Q["User question"] --> INT["intent_llm.py - QueryInterpreter"]
  INT --> DEC["decide_sql"]

  DEC -->|"semantic"| SUM["summarize"]
  DEC -->|"analytic"| SQL["run_sql_path"]
  DEC -->|"blocked or rules"| RET["retrieve_docs"]
```

## 7) SQL analytics subsystem
``` mermaid
flowchart TD
  Q["Question"] --> GATE["sql_orchestrator.py - should_run_sql"]

  GATE -->|"yes"| GEN["llm_sql_agent.py - generate SQL"]
  GEN --> SAN["sanitize and safety checks"]
  SAN --> ENG["sql_engine.py - DuckDB execute"]
  ENG --> RES["rows and aggregates"]

  GATE -->|"no"| RET["retrieve_docs"]
```
## 8) Self-healing loop

``` mermaid
flowchart TD
  SQL["run_sql_path"] --> OK{"SQL success"}
  OK -->|"yes"| SUM["summarize"]
  OK -->|"no"| REF["reflection_node - fix and retry"]
  REF --> DEC["decide_sql"]
  DEC --> SQL
```

## 9) Eval subsystem
``` mermaid
flowchart LR
  QL["eval_questions.jsonl"] --> RUN["run_eval.py"]
  RUN --> OUT["run_outputs - answers"]
  OUT --> SCORE["score_ragas.py"]
  SCORE --> REP["reports - metrics"]
```


## 📊 Evaluation Pipeline (`eval/`)

### `eval_questions.jsonl`
A structured set of evaluation prompts used to benchmark system performance.

---

### `run_eval.py` — Automated Evaluation Runner
Executes evaluation questions end-to-end through the LangGraph pipeline.

---

### `score_ragas.py` — RAGAS Scoring
Computes **RAGAS metrics** (faithfulness, relevance, answer correctness) to quantitatively assess RAG quality.
