# Code/api.py
# Run: uvicorn Code.api:app --reload --host 127.0.0.1 --port 8000

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import os, uuid, shutil

# [UPDATE] Import AppState to ensure type consistency if needed
from .app_langgraph import graph, build_runtime_from_paths, AppState

app = FastAPI(title="BI Agent API", version="1.1")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
UPLOAD_DIR = os.path.abspath(UPLOAD_DIR)
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryResponse(BaseModel):
    final_answer: str
    run_sql: bool = False
    sql_ran: bool = False
    retrieved_chunks: int = 0
    sql: Optional[str] = None
    # [UPDATE] Add this field so the UI receives the JSON chart
    viz_data: Optional[str] = None 

def _save_upload(upload: UploadFile, folder: str) -> str:
    path = os.path.join(folder, upload.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path

@app.post("/query", response_model=QueryResponse)
def query_agent(
    question: str = Form(...),
    csv_file: UploadFile = File(...),
    pdf_files: List[UploadFile] = File(default=[]),
) -> QueryResponse:
    
    # 1) Save files
    session_id = str(uuid.uuid4())[:8]
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    csv_path = _save_upload(csv_file, session_dir)
    pdf_paths = [_save_upload(p, session_dir) for p in (pdf_files or [])]

    # 2) Build runtime
    retriever, con, table_name, type_schema, warnings = build_runtime_from_paths(
        csv_path=csv_path,
        pdf_paths=pdf_paths,
        doc_dir=session_dir,
    )

    # 3) Run LangGraph
    # [UPDATE] Initialize all fields including viz_data
    initial_state: AppState = {
        "question": question,
        "retriever": retriever,
        "con": con,
        "table_name": table_name,
        "type_schema": type_schema,
        "retry_count": 0,
        "error": None,
        "reflection_hint": "",
        "viz_data": None,
        "data_quality_warning": "\n".join(warnings) if warnings else None
    }

    result: Dict[str, Any] = graph.invoke(initial_state)

    sql_output = result.get("sql_output") or {}
    sql_text = sql_output.get("sql") if isinstance(sql_output, dict) else None
    chunks = result.get("retrieved_chunks") or []

    return QueryResponse(
        final_answer=result.get("final_answer", ""),
        run_sql=bool(result.get("run_sql", False)),
        sql_ran=bool(result.get("sql_ran", False)),
        retrieved_chunks=len(chunks) if isinstance(chunks, list) else 0,
        sql=sql_text,
        # [UPDATE] Extract the chart data from the graph result
        viz_data=result.get("viz_data") 
    )