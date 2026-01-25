# qwen2.5:14b  this model is too slow for mac m4 switching to qwen2.5:7b-instruct 

# eval/score_ragas.py
import os
import json
import glob
from typing import Any, Dict, List
from dotenv import load_dotenv

load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig 

# Local Stack
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness

# Config
RUN_DIR = "eval/run_outputs"
OUT_DIR = "eval/reports"

# --- CONFIGURATION: LOCAL 7B MODEL ---
JUDGE_LLM = ChatOllama(
    model="qwen2.5:14b" , # "qwen2.5:7b-instruct",  # Exact model name you pulled
    temperature=0,
    num_ctx=8192,
    timeout=900.0                 #  timeout prevents "Thinking" crashes
)

# --- CONFIGURATION: EMBEDDINGS ---
LOCAL_EMBEDDINGS = OllamaEmbeddings(
    model="nomic-embed-text"      # Requires: ollama pull nomic-embed-text
)

def latest_run_file() -> str:
    files = sorted(glob.glob(os.path.join(RUN_DIR, "run_*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No run_*.jsonl found in {RUN_DIR}")
    return files[-1]

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except FileNotFoundError:
        return []
    return rows

def _clean_contexts(ctx: Any) -> List[str]:
    if not isinstance(ctx, list):
        return []
    cleaned = []
    for c in ctx:
        if isinstance(c, str) and c.strip():
            cleaned.append(c.strip())
        elif isinstance(c, dict):
            text = (c.get("page_content") or c.get("content") or c.get("text") or "")
            if isinstance(text, str) and text.strip():
                cleaned.append(text.strip())
    return cleaned

def main():
    try:
        import langchain_ollama
    except ImportError:
        raise ImportError("Please run: pip install langchain-ollama")

    os.makedirs(OUT_DIR, exist_ok=True)
    run_path = latest_run_file()
    print(f"📂 Loading run: {run_path}")
    
    run_rows = load_jsonl(run_path)
    ragas_rows = []

    for r in run_rows:
        qid = r.get("id")
        q = (r.get("question") or "").strip()
        a = (r.get("answer") or "").strip()
        ctx = _clean_contexts(r.get("contexts", []))
        gt_text = str(r.get("reference") or r.get("ground_truth") or "").strip()

        if not q or not a or len(ctx) == 0:
            continue

        ragas_rows.append({
            "id": qid,
            "question": q,
            "answer": a,
            "contexts": ctx,
            "ground_truth": gt_text,
        })

    if not ragas_rows:
        raise ValueError("No valid rows to score.")

    print(f"🚀 Scoring {len(ragas_rows)} items using {JUDGE_LLM.model}...")
    print("⚡️ This will be fast (Sequential Processing).")

    ds = Dataset.from_list(ragas_rows)

    metrics_list = [
        Faithfulness(),
        AnswerRelevancy(),
        AnswerCorrectness()
    ]

    # FORCE SEQUENTIAL EXECUTION (max_workers=1)
    # This is critical for the MacBook Air to prevent overheating/timeouts.
    result = evaluate(
        ds,
        metrics=metrics_list,
        llm=JUDGE_LLM,
        embeddings=LOCAL_EMBEDDINGS,
        run_config=RunConfig(timeout=900,max_workers=1) 
    )
    
    df = result.to_pandas()

    base = os.path.basename(run_path).replace(".jsonl", "")
    out_csv = os.path.join(OUT_DIR, f"{base}_ragas_scores_local.csv")
    df.to_csv(out_csv, index=False)

    summary = {
        "run_file": run_path,
        "rows_scored": len(df),
        "judge_model": JUDGE_LLM.model,
        "metrics": {
            k: float(df[k].mean()) for k in ["faithfulness", "answer_relevancy", "answer_correctness"] if k in df
        }
    }

    out_json = os.path.join(OUT_DIR, f"{base}_ragas_summary_local.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved CSV: {out_csv}")
    print("\n📊 LOCAL SUMMARY:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()