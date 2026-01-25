# eval/run_eval.py
import json
import os
from datetime import datetime
from typing import Any, Dict, List

# Ensure these imports match your project structure
from Code.app_langgraph import graph, build_runtime, AppState

OUT_DIR = "eval/run_outputs"

def chunk_texts(result: Dict[str, Any]) -> List[str]:
    """Extract text from retrieved document chunks."""
    chunks = result.get("retrieved_chunks") or []
    texts: List[str] = []
    for c in chunks:
        # Handle both LangChain Document objects and dicts
        t = getattr(c, "page_content", None) or c.get("page_content") or c.get("text")
        if isinstance(t, str) and t.strip():
            texts.append(t.strip())
    return texts

def format_sql_evidence(result: Dict[str, Any]) -> str:
    """
    Robust extraction: Checks both the top-level state AND the summary_payload
    to find SQL evidence, even if the state was cleaned up.
    """
    # 1. Try fetching from top-level state
    sql_info = result.get("sql_output")
    
    # 2. If missing (because it was cleaned up), check summary_payload
    if not sql_info:
        payload = result.get("summary_payload") or {}
        sql_info = payload.get("sql_output")
        
    # Ensure sql_info is a dictionary (handle None)
    sql_info = sql_info or {}

    # 3. Format the evidence
    if sql_info.get("sql_ran") and sql_info.get("sql_result"):
        query = sql_info.get("sql", "UNKNOWN QUERY")
        data = sql_info.get("sql_result", "NO DATA")
        
        evidence = (
            f"### SQL CONTEXT ###\n"
            f"THE AGENT RAN THIS QUERY: {query}\n"
            f"THE DATABASE RETURNED THIS DATA: {data}\n"
            f"### END SQL CONTEXT ###"
        )
        return evidence
    
    return ""

def run_eval(eval_path: str = "eval/eval_questions.jsonl"):
    os.makedirs(OUT_DIR, exist_ok=True)

    print("🚀 Initializing Runtime (Ingestion + DB)...")
    retriever, con, table_name, type_schema = build_runtime()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_DIR, f"run_{run_id}.jsonl")
    
    print(f"📂 Output will be saved to: {out_path}")
    print("⏳ Starting evaluation loop...")

    with open(eval_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            # Allow comment lines
            if line.startswith("//"):
                continue

            item = json.loads(line)

            qid = item.get("id")
            question = item.get("question", "")
            reference = item.get("reference") or item.get("ground_truth") 

            if not question.strip():
                continue

            print(f"   Processing Q: {qid} - {question[:40]}...")

            initial_state: AppState = {
                "question": question,
                "retriever": retriever,
                "con": con,
                "table_name": table_name,
                "type_schema": type_schema,
            }

            # 1. Run the Agent
            result: Dict[str, Any] = graph.invoke(initial_state)

            # 2. Extract Document Contexts
            contexts = chunk_texts(result)

            # 3. CRITICAL FIX: Append SQL Evidence
            # This ensures Ragas sees the numbers the agent saw.
            sql_evidence = format_sql_evidence(result)
            if sql_evidence:
                contexts.append(sql_evidence)

            # 4. Extract SQL metadata for the log
            sql_output = result.get("sql_output") or {}
            sql_text = sql_output.get("sql") if isinstance(sql_output, dict) else None

            record = {
                "id": qid,
                "question": question,
                "reference": reference, 
                "expected_route": item.get("expected_route"),

                "pred_run_sql": bool(result.get("run_sql", False)),
                "sql_ran": bool(result.get("sql_ran", False)),
                "sql": sql_text,

                "answer": result.get("final_answer", ""),
                "contexts": contexts, # Now includes Docs + SQL Evidence
                "retrieved_k": len(contexts),
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Eval run finished! File saved: {out_path}")
    print(f"👉 Next Step: Run 'python eval/score_ragas.py' to score this run.")

if __name__ == "__main__":
    run_eval()