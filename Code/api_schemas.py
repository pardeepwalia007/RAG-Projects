from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
"""
It exposes your LangGraph agent as a callable service that any UI (web, mobile, internal tool) can talk to — without touching your core logic.
"""
# define the style for request and response
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question for the BI agent")
    # Optional overrides (nice for demos + future generalization)
    csv_path: Optional[str] = Field(None, description="Optional CSV path override")
    pdf_paths: Optional[List[str]] = Field(None, description="Optional PDF paths override")
    top_k: int = Field(4, ge=1, le=20, description="Retriever top-k chunks")


class AskResponse(BaseModel):
    question: str
    final_answer: str

    # Debug/telemetry 
    run_sql: bool = False
    sql_ran: bool = False
    sql: Optional[str] = None
    row_count: Optional[int] = None

    retrieved_k: int = 0
    source_type: str = "Docs Only"  # or "Hybrid (Docs + SQL)"

    # Optional: return these only if you want
    intent_spec: Optional[Any] = None
    errors: Optional[List[str]] = None