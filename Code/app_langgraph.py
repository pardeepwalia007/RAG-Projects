# implementation of app.py using langgraph with Invisible Reflection Hints
import re

from duckdb import df
from Code.ingestion import ingest_files
from Code.sql_engine import load_csv_to_duckdb
from Code.pdf_to_markdown import pdfs_to_markdown
from Code.vectorize import build_retriever
from Code.sql_orchestrator import should_run_sql
from Code.summarization_agent import summarize_with_llama
from Code.tests_logger import log_test
from Code.llm_sql_agent import sql_pipeline_structured
import logging
from Code.intent_llm import QueryInterpreter
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Any, Optional, List, Dict, Tuple
import pandas as pd
import plotly.express as px
import plotly.io as pio
import json

# Silencing noisy loggers to keep terminal output clean
NOISY_LOGGERS = ["httpx", "urllib3", "ollama", "chromadb", "langchain", "langchain_core"]
for logger in NOISY_LOGGERS:
    logging.getLogger(logger).setLevel(logging.CRITICAL)
logging.disable(logging.CRITICAL)

# Shared state
class AppState(TypedDict, total=False):
    question: str
    paths: List[str]
    csv_path: str
    pdf_paths: List[str]
    paths_md: List[str]
    doc_dir: str
    con: Any
    table_name: str
    type_schema: Dict[str, Any]
    num_cols: List[str]
    retriever: Any
    retrieved_chunks: List[Any]
    doc_evidence: str
    run_sql: bool
    intent_spec: Any
    sql_ran: bool
    sql_output: Optional[Dict[str, Any]]
    summary_payload: Dict[str, Any]
    final_answer: str
    # NEW: Self-Correction Fields
    error: Optional[str]
    retry_count: int
    reflection_hint: str # Used to fix SQL without cluttering the summary
    viz_data: Optional[str]  # New field for visualization data
    data_quality_warning: Optional[str]  # New field for data quality warnings
    viz_method: Optional[str]  # New field for visualization method description
# Nodes

def retrieve_docs(state: AppState) -> AppState:
    """Retrieves relevant document chunks for the query."""
    print("\n [Langraph] Node: retrieve_docs")
    q = state["question"]
    retriever = state["retriever"]
    print("\n [Langraph] Node: retriever_docs")
    chunks = retriever.invoke(q)
    doc_evidence = "\n".join(c.page_content for c in chunks) if chunks else ""
    print(f"[Langraph] Retrieved chunks: {len(chunks) if chunks else 0}")
    print(f"\ntype_schema: {state['type_schema']}")
    return {
        "retrieved_chunks": chunks,
        "doc_evidence": doc_evidence,
    }

def decide_sql(state: AppState) -> AppState:
    """Generalized decision node that uses dynamic schema and error feedback."""
    q = state["question"]
    hint = ""
    print("\n [Langraph] Node: decide_sql")
    # If we are in a retry loop, we generate a dynamic reflection hint
    if state.get("retry_count", 0) > 0:
        last_err = state.get("error", "Unknown SQL Error")
        # We inject the REAL schema and the REAL error back into the prompt
        # without hardcoding any specific column names.
        hint = (
            f"\n\n[SELF-CORRECTION HINT]: Your previous SQL generated an error: '{last_err}'.\n"
            f"1. LOOK at the actual table columns: {list(state['type_schema'].keys())}.\n"
            f"2. READ the 'doc_evidence' chunks carefully to find how the user's requested metric "
            f"maps to these specific columns.\n"
            f"3. RE-PLAN your query based ONLY on the columns listed above and the rules in the docs."
        )

    # Use the combined text to decide if we should proceed
    run_sql = should_run_sql(q + hint) 

    return {
        "run_sql": run_sql,
        "reflection_hint": hint 
    }

def route_after_gate(state: AppState) -> str:
    return "run_sql_path" if state.get("run_sql") else "summarize"
# updated function with reflection-aware intent refinement
def run_sql_path(state: AppState) -> AppState:
    """Runs SQL pipeline with Intent Patching (keeps 'q' pure)."""
    print("\n [Langraph] Node: run_sql_path")

    q = state["question"]
    doc_evidence = state.get("doc_evidence", "") or ""

    # 1. Standard Setup
    is_retry = state.get("retry_count", 0) > 0
    retry_context = state.get("reflection_hint", "") if is_retry else None
    con = state["con"]
    table_name = state["table_name"]
    type_schema = state["type_schema"]

    # 2. Run Interpreter (Standard)
    try:
        interpreter = QueryInterpreter(con, table_name, type_schema)
        refined_spec = interpreter.refine_intent(
            q,
            business_context=doc_evidence,
            retry_context=retry_context
        )
    except Exception as e:
        error_msg = f"Intent Refinement Error: {str(e)}"
        print(f"⚠️ {error_msg}")
        return {
            "sql_ran": False,
            "error": error_msg,
            "sql_output": {"error": error_msg},
        }

    # 3. Run SQL Pipeline (SQL agent now returns data_quality_warning)
    output = sql_pipeline_structured(
        q,
        refined_spec,
        con=con,
        table_name=table_name,
        type_schema=type_schema,
    )

    print(f"-----Sql_output----: {output.get('sql')}")
    print(f"-----Refined-Intent----: {refined_spec}")

    # 4. Output Handling (Standard)
    current_error = output.get("error")
    sql_evidence = ""
    if output.get("sql_ran") and output.get("sql_result"):
        res = output["sql_result"]
        rows = res.get("rows", [])
        cols = res.get("columns", [])
        if rows:
            sql_evidence += "\n\n[SQL_RESULT]\n"
            sql_evidence += f"COLUMNS: {', '.join(str(c) for c in cols)}\n"
            preview_rows = rows[:20]
            for r in preview_rows:
                sql_evidence += f"ROW: {', '.join(str(x) for x in r)}\n"

    dq_warn = output.get("data_quality_warning") or state.get("data_quality_warning")

    return {
        "intent_spec": refined_spec,
        "sql_ran": bool(output.get("sql_ran")),
        "sql_output": output,
        "doc_evidence": doc_evidence,
        "error": current_error,
        "data_quality_warning": dq_warn,
        "sql_evidence": sql_evidence,
    }

def reflection_node(state: AppState) -> AppState:
    new_count = state.get("retry_count", 0) + 1
    print(f"🔄 [LangGraph] Node: Reflection (Self-Correction Attempt {new_count})")
    return {"retry_count": new_count}

def route_after_sql(state: AppState) -> str:
    print("\n [Langraph] Routing after SQL Execution...")
    if state.get("error") and state.get("retry_count", 0) < 3:
        print(f"❌ SQL Execution Failed. Re-routing for autonomous correction...")
        return "reflection_node"
    
    # [FIX] Return 'visualize_data' so it matches the edge definition!
    return "visualize_data"

def visualize_data(state: AppState) -> AppState:
    """
    Analyzes SQL output, generates Plotly JSON, and flags data quality issues.
    """
    viz_method = None
    print("--- CHECKING FOR VISUALIZATION ---")
    quality_warning = None # [NEW] Initialize warning
    prior_warning = state.get("data_quality_warning")

    # [GUARDRAIL 1] Check Intent: Metadata queries (listing columns) need no charts.
    intent_spec = state.get("intent_spec")
    current_intent = getattr(intent_spec, "intent", "") if intent_spec else ""
    
    if current_intent == "metadata":
        print("--- VIZ SKIP: Metadata intent detected ---")
        return {"viz_data": None, "data_quality_warning": prior_warning}

    try:
        # 1. Correct Data Extraction
        sql_output = state.get("sql_output", {})
        res = sql_output.get("sql_result") if sql_output.get("sql_result") else sql_output
        rows = res.get("rows", [])
        columns = res.get("columns", [])

        # Even if rows is 0, we check if the columns involved are known to be "dirty"
        if not rows or len(rows) == 0:
            # If the user was looking for 'rating' and we got 0 rows, 
            # it's almost certain the '|' character caused the SQL crash/empty set.
            if any("rating" in col.lower() for col in columns):
                quality_warning = "Data Quality Alert: The analysis returned no results. This is likely due to non-numeric values (like '|') in the 'rating' column preventing the calculation. Cleaning the source data is recommended."
            
            print("--- VIZ SKIP: No rows found, but quality check performed ---")
            return {"viz_data": None, "data_quality_warning": quality_warning or prior_warning}

        # 2. Convert to DataFrame
        df = pd.DataFrame(rows, columns=columns)

        # [GUARDRAIL 2] Minimum Row Check
        # A chart with 1 row (John Smith) is useless. We need at least 2 rows to compare.
        if len(df) < 2:
            print(f"--- VIZ SKIP: Only {len(df)} row(s) found. Need 2+ for a chart. ---")
            return {"viz_data": None, "data_quality_warning": quality_warning or prior_warning}

        # Helper: Detect column types
        def is_date(col):
            return pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower() or "year" in col.lower() or "month" in col.lower()
        
        def is_number_robust(col):
            """Detects if a column is mostly numeric and flags garbage values."""
            nonlocal quality_warning
            s = pd.to_numeric(df[col], errors='coerce')
            valid_count = s.notnull().sum()
            total = len(s)
            
            # [DATA QUALITY ALERT] If we find some numbers but also some garbage
            if 0 < valid_count < total:
                garbage_count = total - valid_count
                quality_warning = f"Data Quality Alert: {garbage_count} non-numeric values (e.g. '|') were excluded from the '{col}' analysis."
            
            # Return true if >90% of column is numeric
            return (valid_count / total) > 0.9

        # TRIGGER 2: Heuristics
        fig = None
        date_cols = [c for c in columns if is_date(c)]
        num_cols = [c for c in columns if is_number_robust(c) and c not in date_cols]
        
        # Case A: Time Series (Date + Number)
        if date_cols and num_cols:
            x_col = date_cols[0]
            y_col = num_cols[0]
            # [FIX] Clean data for plotting
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            df = df.dropna(subset=[y_col]).sort_values(by=x_col)
            fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}", markers=True)
            viz_method = f"Line chart of '{y_col}' over '{x_col}'."

        # Case B: Ranking (Category + Number)
        elif len(columns) >= 2:
            text_cols = [c for c in columns if c not in num_cols and c not in date_cols]
            if text_cols and num_cols:
                x_col = text_cols[0]
                y_col = num_cols[0]
                # [FIX] Clean data for plotting
                df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
                df = df.dropna(subset=[y_col])
                
                if len(df) > 10:
                    fig = px.bar(df, x=y_col, y=x_col, orientation='h', title=f"{y_col} by {x_col}")
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                else:
                    fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                viz_method = f"Bar chart of '{y_col}' by '{x_col}'."
        # Case C: Distribution (Single Numeric Column)
        elif len(columns) == 1:
            col = columns[0]
            if is_number_robust(col):
                # [FIX] Force numeric conversion and drop garbage
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=[col])
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                fig.update_layout(bargap=0.1)
                viz_method = f"Histogram distribution of '{col}'."
        # Case D: Relationship (Scatter Plot) - NEW
        # Triggered if intent is 'relationship' OR (2 numeric cols + high cardinality)
        elif (current_intent in ["relationship","visualization"]) and len(num_cols) >= 2:
            x_col = num_cols[1] # Independent
            y_col = num_cols[0] # Dependent
            
            # Scatter plots need raw data, not aggregations usually
            df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            df = df.dropna(subset=[x_col, y_col])
            
            if len(df) > 2:
                fig = px.scatter(df, x=x_col, y=y_col, title=f"Relationship: {y_col} vs {x_col}", trendline="ols")
                viz_method = f"Scatter plot of '{y_col}' vs '{x_col}'."

        # 3. Return viz and the quality warning
        if fig:
            print(f"--- VIZ GENERATED: {type(fig)} ---")
            return {
                "viz_data": pio.to_json(fig),
                "viz_method": viz_method,
                "data_quality_warning": quality_warning or prior_warning # [NEW] Passed to next node
            }

        return {"viz_data": None, "viz_method": viz_method, "data_quality_warning": quality_warning or prior_warning}

    except Exception as e:
        print(f"⚠️ VIZ ERROR: {e}")
        return {"viz_data": None, "data_quality_warning": None}

def summarize(state: AppState) -> AppState:
    """Produces final response using clean doc_evidence."""
    print("\n [Langraph] Node: summarize")
    q = state["question"]
    doc_evidence = state.get("doc_evidence")
    run_sql = state.get("run_sql", False)
    sql_output = state.get("sql_output") if run_sql else None

    # [FIX] Initialize variable to empty string to prevent UnboundLocalError
    viz_context = "" 
    
    viz_data = state.get("viz_data")
    has_viz = bool(viz_data)
    
    # Now this line is safe
    augmented_evidence = doc_evidence + viz_context

    dq = state.get("data_quality_warning")
    if dq:
        augmented_evidence = augmented_evidence + f"\n\n[DATA_QUALITY_ALERT]\n{dq}\n"
    
    intent_spec = state.get("intent_spec")
    intent_val = getattr(intent_spec, "intent", None)

    summary_payload = {
        "question": q,
        "source_type": "Hybrid (Docs + SQL)" if run_sql else "Docs Only",
        "business_rules": augmented_evidence,
        "sql_output": sql_output,
        "data_quality_warning": state.get("data_quality_warning"),  # [NEW] Pass warning to summarizer
        "intent": intent_val,
        "viz_data": has_viz,
        "viz_method": state.get("viz_method")  # [NEW] Pass viz method description
    }

    final_answer = summarize_with_llama(
        question=q,
        evidence=summary_payload,
        source_type=summary_payload["source_type"],
    )

    log_test(q, final_answer)

    return {
        "summary_payload": summary_payload,
        "final_answer": final_answer,
        "sql_output": None,
        "reflection_hint": "" 
    }

# Ingestion & Build Runtime Logic
def build_runtime() -> Tuple[Any, Any, str, Dict[str, Any]]:
    paths = [
        r"/Users/pardeepwalia/Desktop/Data/Agentic_RAG/Data/docs/Business_Metrics_Detailed.pdf",
        r"/Users/pardeepwalia/Desktop/Data/Agentic_RAG/Data/csv/Sales_enriched.csv",
        r"/Users/pardeepwalia/Desktop/Data/Agentic_RAG/Data/docs/Business_Rules_Detailed.pdf",
    ]
    csv_path, pdf_paths = ingest_files(paths)
    con, table_name, type_schema, num_cols,warnings = load_csv_to_duckdb(csv_path)
    paths_md, errors_md, is_md = pdfs_to_markdown(pdf_paths, r"/Users/pardeepwalia/Desktop/Data/Agentic_RAG/Data/docs/")
    retriever = build_retriever(paths_md)
    return retriever, con, table_name, type_schema,warnings

def build_runtime_from_paths(csv_path: str, pdf_paths: List[str], doc_dir: str) -> Tuple[Any, Any, str, Dict[str, Any]]:
    con, table_name, type_schema, num_cols,warnings = load_csv_to_duckdb(csv_path)
    paths_md, errors_md, is_md = pdfs_to_markdown(pdf_paths, doc_dir)
    retriever = build_retriever(paths_md) if paths_md else None
    return retriever, con, table_name, type_schema,warnings

# Graph Construction
graph_builder = StateGraph(AppState)
graph_builder.add_node("run_sql_path", run_sql_path)
graph_builder.add_node("reflection_node", reflection_node)
graph_builder.add_node("summarize", summarize)
graph_builder.add_node("retrieve_docs", retrieve_docs)
graph_builder.add_node("decide_sql", decide_sql)
graph_builder.add_node("visualize_data", visualize_data)

graph_builder.add_edge(START, "retrieve_docs")
graph_builder.add_edge("retrieve_docs", "decide_sql")

graph_builder.add_conditional_edges(
    "decide_sql",
    route_after_gate,
    {"run_sql_path": "run_sql_path", "summarize": "summarize"}
)

graph_builder.add_conditional_edges(
    "run_sql_path",
    route_after_sql,
    {
        "reflection_node": "reflection_node", 
        "visualize_data": "visualize_data"  # <--- CHANGED: Route success to Viz, changed the path of sumarizer to viz
    }
)

#  Reflection Loop
graph_builder.add_edge("reflection_node", "decide_sql")

# Visualization -> Summarization
# Once the chart is generated (or skipped), go to summary
graph_builder.add_edge("visualize_data", "summarize")

graph_builder.add_edge("summarize", END)

graph = graph_builder.compile()

# Export Mermaid Diagram
png_bytes = graph.get_graph().draw_mermaid_png()
with open("agentic_rag_langgraph.png", "wb") as f:
    f.write(png_bytes)

# Main Loop CLI
def bi_agent():
    retriever, con, table_name, type_schema, warnings = build_runtime()
    # Prepare warning message
    dq_msg = "\n".join(warnings) if warnings else None
    if dq_msg:
        print(f"\n[SYSTEM ALERT]: {dq_msg}\n")
   # CLI Loop
    while True:
        q = input("User:  ")
        if not q or q.lower() in {"exit", "quit", "q"}:
            break
        initial_state: AppState = {
            "question": q,
            "retriever": retriever,
            "con": con,
            "table_name": table_name,
            "type_schema": type_schema,
            "retry_count": 0,
            "error": None,
            "reflection_hint": "",
            "data_quality_warning": dq_msg
        }
        result = graph.invoke(initial_state)
        print("\n🤖 Agent Response:")
        print(result["final_answer"])
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    bi_agent()

