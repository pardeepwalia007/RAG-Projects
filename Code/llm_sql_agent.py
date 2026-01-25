# Generates and executes SQL queries from interpreted specs.
from __future__ import annotations
import json
import re
from typing import Dict, Any, Optional, Tuple, Set, List

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from .sql_orchestrator import (
    validate_sql,
    enforce_safety_limits,
    enforce_transaction_semantics,
    enforce_revenue_semantics,
    enforce_ranking_shape,
    enforce_distribution_shape
)

class QuerySpec(BaseModel):
    intent: str = Field(description="The action: 'ranking', 'aggregation','filter', or distribution")
    metric: Optional[str] = Field(default=None, description="The numeric column name or function from the schema")
    # Changed to entity_columns (plural) List[str] to match query_refiner.py
    entity_columns: List[str] = Field(
        default_factory=list, 
        description="List of group-by column names from the schema"
    )
    filter_value: Optional[str] = Field(default=None, description="The specific value found in DB samples, or null")
    refined_instruction: str = Field(description="A clean technical instruction for a SQL coder")


SQL_MODEL = "qwen2.5:14b-instruct-q5_K_M" #"qwen3-coder:30b"#"qwen2.5:14b" # prev qwen2.5:7b-instruct
llm = ChatOllama(model=SQL_MODEL, temperature=0)

# never chage the sanitization logic; it ensures we get valid SQL statements
def _sanitize_sql(raw_sql: str) -> str:
    """
    Strictly extracts the SQL statement by ignoring any leading/trailing LLM prose.
    Fixed: Uses a robust regex to find the start of the actual query.
    """
    if not raw_sql:
        return ""

    # 1. Strip markdown code blocks
    s = raw_sql.replace("```sql", "").replace("```SQL", "").replace("```", "").strip()
    
    # 2. Extract starting from the first SELECT or WITH (case-insensitive)
    # This ignores phrases like "Here is your query:" or "Assistant: "
    match = re.search(r"(?i)\b(SELECT|WITH)\b.*", s, re.DOTALL)
    if match:
        s = match.group(0).strip()

    # 3. Take only the first statement if multiple exist
    s = s.split(";")[0].strip()
    
    if s and not s.endswith(";"):
        s += ";"
        
    return s

def _extract_quoted_identifiers(sql: str) -> Set[str]:
    """Extracts column names quoted in SQL for validation."""
    return set(re.findall(r'"([^"]+)"', sql or ""))

def _split_csvish(value: Any) -> List[str]:
    """Converts schema strings to lists of columns."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, (set, tuple)):
        return [str(x).strip() for x in list(value) if str(x).strip()]
    s = str(value).strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]

def _extract_known_columns(type_schema: Dict[str, Any]) -> Set[str]:
    """Extracts all known column names from schema."""
    meta_keys = {"TABLE", "PRIMARY_KEY_ID", "NUMERIC COLUMNS", "TEXT COLUMNS", "DATE COLUMNS"}
    keys = set(type_schema.keys())
    if keys.intersection(meta_keys):
        numeric_cols = _split_csvish(type_schema.get("NUMERIC COLUMNS"))
        text_cols = _split_csvish(type_schema.get("TEXT COLUMNS"))
        date_cols = _split_csvish(type_schema.get("DATE COLUMNS"))
        pk = str(type_schema.get("PRIMARY_KEY_ID") or "").strip()
        cols = set(numeric_cols + text_cols + date_cols)
        if pk:
            cols.add(pk)
        return cols
    return set(type_schema.keys())

def _get_pk_from_schema(type_schema: Dict[str, Any]) -> Optional[str]:
    """Retrieves primary key from schema."""
    pk = str(type_schema.get("PRIMARY_KEY_ID") or "").strip()
    return pk or None

def _schema_column_precheck(sql: str, table_name: str, type_schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validates that SQL uses only known schema columns."""
    known = _extract_known_columns(type_schema)
    used = _extract_quoted_identifiers(sql)
    used = {u for u in used if u != table_name}
    unknown = sorted([u for u in used if u not in known])
    if unknown:
        return False, f"Unknown column(s) referenced: {unknown}. Known columns include: {sorted(list(known))[:20]}..."
    return True, None

def _force_scalar_shape(sql: str) -> str:
    """Removes GROUP BY for scalar aggregations."""
    s = (sql or "").strip().rstrip(";")
    s = re.sub(r"\bGROUP\s+BY\b.*?(?=(ORDER\s+BY|LIMIT|$))", "", s, flags=re.IGNORECASE | re.DOTALL).strip()
    return f"{s};"

def _coerce_numeric_where_clauses(sql: str, type_schema: Dict[str, Any]) -> str:
    """
    If WHERE compares a TEXT column with a number (>, <, >=, <=),
    rewrite to TRY_CAST(regexp_replace(col) AS DOUBLE) <op> number.
    Dataset-agnostic: uses schema TEXT COLUMNS list only.
    """
    text_cols = _split_csvish(type_schema.get("TEXT COLUMNS"))

    def _cast_expr(col: str) -> str:
        cleaned = f"regexp_replace(\"{col}\", '[^0-9\\.\\-]+', '', 'g')"
        return f"TRY_CAST({cleaned} AS DOUBLE)"

    out = sql

    # Handles patterns like: WHERE "col" > 50   / AND "col" <= 12.5
    for col in text_cols:
        pattern = rf'(" {col} "|"{col}")\s*(>=|<=|>|<)\s*([0-9]+(?:\.[0-9]+)?)'
        def repl(m):
            op = m.group(2)
            num = m.group(3)
            return f"{_cast_expr(col)} {op} {num}"

        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)

    return out

def _wants_percentage(user_query: str) -> bool:
    q = (user_query or "").lower()
    return any(tok in q for tok in ["percent", "percentage", "rate", "ratio", "share", "proportion", "%"])

def _add_pct_column_if_grouped(sql: str) -> str:
    """
    If SQL is grouped and has an aggregate alias, inject:
      ROUND(alias * 100.0 / NULLIF(SUM(alias) OVER (), 0), 2) AS pct
    This is dataset-agnostic and does NOT need column names.
    """
    s = (sql or "").strip().rstrip(";")

    # Only for grouped outputs
    if re.search(r"\bGROUP\s+BY\b", s, flags=re.IGNORECASE) is None:
        return sql

    # Find a likely aggregate alias (last AS <alias> in SELECT list)
    # Ex: COUNT(*) AS ct, SUM("money") AS total_revenue
    aliases = re.findall(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)\b", s, flags=re.IGNORECASE)
    if not aliases:
        return sql
    value_alias = aliases[-1]

    # Avoid double-injecting
    if re.search(r"\bAS\s+pct\b", s, flags=re.IGNORECASE):
        return sql

    # Inject pct after that alias occurrence (first hit is safest)
    s2 = re.sub(
        rf"(\bAS\s+{re.escape(value_alias)}\b)",
        rf"\1, ROUND({value_alias} * 100.0 / NULLIF(SUM({value_alias}) OVER (), 0), 2) AS pct",
        s,
        count=1,
        flags=re.IGNORECASE
    )
    return s2 + ";"



def generate_structured_sql(interpreter_spec: QuerySpec, table_name: str, type_schema_str: str) -> str:
    """Generates SQL from QuerySpec using LLM with schema context."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a DuckDB SQL Expert. 
        TASK: Generate a valid DuckDB SQL query based on the user's specification and the provided schema.
        CONTEXT:
        - Table: {table_name}
        - Schema: {schema}

        CRITICAL RULES:
        1. SCHEMA FIDELITY: You MUST use the EXACT casing of columns as shown in the schema.
           - If schema says "Product_Name", DO NOT write "product_name".
           - Always use DOUBLE QUOTES for columns: "Product_Name".
        2. TEXT MATCHING: Always use 'ILIKE' for text filters to be case-insensitive.
           - Correct: WHERE "Product_Name" ILIKE '%macbook%'
         2.5 CASTING / DIRTY-NUMERIC RULE (MANDATORY):
            - NEVER use CAST(...) for numeric conversion on columns that may contain non-numeric characters.
             - ALWAYS use TRY_CAST(... AS DOUBLE) instead.
        3. DATES (ABSOLUTE): 
           - CONTINUOUS RANGES: DO NOT use multiple OR statements for continuous years (e.g., 2021, 2022, 2023).
           - REQUIRED FORMAT: You MUST use a single continuous range.
             * BAD: ... WHERE (date >= '2021...') OR (date >= '2022...') ...
             * GOOD: ... WHERE "sale_date" >= '2021-01-01' AND "sale_date" < '2024-01-01' ...
           - LOGIC SAFETY: If you ever use 'OR', you MUST wrap the conditions in parentheses if an 'AND' follows.
             * FATAL ERROR: WHERE Year=2021 OR Year=2022 AND Cat='A' (Returns ALL of 2021).
             * CORRECT: WHERE (Year=2021 OR Year=2022) AND Cat='A'.
           - NO TEXT MATCHING: NEVER use 'ILIKE' or 'partial match' for dates.
        4. GROUPING LOGIC: 
           - You MUST repeat the calculation in GROUP BY.
           - Correct: SELECT strftime("sale_date", '%Y-%m') ... GROUP BY strftime("sale_date", '%Y-%m')
        5. COMPARISONS: 
           - Retrieve ALL months for the requested years. 
           - NEVER use 'LIMIT 1' for trend or comparison queries.

        CRITICAL OUTPUT FORMAT:
        - OUTPUT THE SQL ONLY. 
        - DO NOT include any introductory text, explanations, or conversational filler.
        - DO NOT include "Here is the query" or "Sure, I can help".
        - If you fail to follow this, the system will crash.
         

        FEW-SHOT PATTERNS:

        -- Pattern 1: Trend (Standard)
        User: "Sales trend 2023"
        Assistant: SELECT strftime("sale_date", '%Y-%m') AS month, SUM(revenue) FROM {table_name} WHERE year("sale_date") = 2023 GROUP BY strftime("sale_date", '%Y-%m') ORDER BY month LIMIT 1000;

        -- Pattern 2: Top Products
        User: "Top 5 products"
        Assistant: SELECT "Product_Name", SUM(revenue) AS total FROM {table_name} GROUP BY "Product_Name" ORDER BY total DESC LIMIT 5;

        -- Pattern 3: Partial Match
        User: "Total sales for Macbook"
        Assistant: SELECT SUM(revenue) FROM {table_name} WHERE "Product_Name" ILIKE '%Macbook%';
        """),
        ("human", """SPECIFICATION:
        - Intent: {intent}
        - Metric: {metric}
        - Entities: {entities}
        - Filter: {filter_val}
        - Refined Instruction: {goal}

        Generate SQL.

         NEVER USE OTHER SQL DIALECTS OR FORMATTING. USE ONLY DUCKDB SQL SYNTAX.  
         
         """)
    ])

    # Convert list of entities to a comma-separated string for the prompt
    entity_str = ", ".join([f'"{e}"' for e in interpreter_spec.entity_columns]) if interpreter_spec.entity_columns else "None"

    chain = prompt | llm
    response = chain.invoke({
        "schema": type_schema_str,
        "table_name": table_name,
        "intent": interpreter_spec.intent,
        "metric": interpreter_spec.metric,
        "entities": entity_str,
        "filter_val": interpreter_spec.filter_value if interpreter_spec.filter_value else "None",
        "goal": interpreter_spec.refined_instruction
    })
    return (response.content or "").strip()

def sql_pipeline_structured(
    user_query: str,
    interpreter_spec: QuerySpec,
    con,
    table_name: str,
    type_schema: dict,
    limit: int = 1000
) -> Dict[str, Any]:
    """Orchestrates SQL generation, validation, and execution."""

    try:
        type_schema_str = json.dumps(type_schema, indent=2, default=str)
    except Exception:
        type_schema_str = str(type_schema)
    # Generate initial SQL
    raw_sql = generate_structured_sql(interpreter_spec, table_name, type_schema_str)
    # Sanitize SQL
    final_sql = _sanitize_sql(raw_sql)
    # Enforce distribution shape if needed
    if interpreter_spec.intent == "distribution":
        metric_col = interpreter_spec.metric

        # ✅ if metric missing but entities exist (older specs), use first entity
        if (not metric_col) and getattr(interpreter_spec, "entity_columns", None):
            if interpreter_spec.entity_columns:
                metric_col = interpreter_spec.entity_columns[0]

        metric_s = str(metric_col or "")
        if (not metric_col) or ("(" in metric_s and "TRY_CAST" not in metric_s.upper()):
            numeric_cols = type_schema.get("NUMERIC COLUMNS", [])
            if isinstance(numeric_cols, str):
                numeric_cols = [c.strip() for c in numeric_cols.split(",") if c.strip()]
            metric_col = numeric_cols[0] if numeric_cols else None

        if not metric_col:
            return {
                "sql_ran": False,
                "sql": None,
                "sql_result": None,
                "error": "Distribution intent needs a target column, but none was found."
            }

        # Enforce distribution shape
        final_sql = enforce_distribution_shape(final_sql, table_name, metric_col, limit=limit)


    # Enforce scalar shape if entity_columns list is empty
    if interpreter_spec.intent == "aggregation" and not interpreter_spec.entity_columns:
        final_sql = _force_scalar_shape(final_sql)

    

    # Generalized Fix: Check existence first
    if "revenue" in type_schema.get("NUMERIC COLUMNS", []):
         final_sql = enforce_revenue_semantics(user_query, final_sql, revenue_col="revenue")
    final_sql = enforce_ranking_shape(user_query, final_sql, default_limit=5, intent=interpreter_spec.intent)
    final_sql = enforce_safety_limits(user_query, final_sql, max_limit=limit, intent=interpreter_spec.intent)
    # Enforce transaction semantics if PK exists
    pk = _get_pk_from_schema(type_schema)
    if pk:
        final_sql = enforce_transaction_semantics(user_query, final_sql, pk)
    # Coerce numeric WHERE clauses
    final_sql = _coerce_numeric_where_clauses(final_sql, type_schema) 
    cast_warn = None
    if re.search(r"\bTRY_CAST\s*\(", final_sql, flags=re.IGNORECASE):
        # collect quoted column names that appear inside TRY_CAST(...)
        cols = set(re.findall(r'TRY_CAST\s*\(.*?"([^"]+)".*?\)', final_sql, flags=re.IGNORECASE | re.DOTALL))

        if cols:
            col_list = ", ".join(sorted(cols))
            cast_warn = (
                f"Data Quality Alert: Numeric type conversion (TRY_CAST) was applied on: {col_list}. "
                "Results may exclude non-numeric values; clean the source data to avoid misleading analysis."
            )
        else:
            cast_warn = (
                "Data Quality Alert: Numeric type conversion (TRY_CAST) was applied. "
                "Results may exclude non-numeric values; clean the source data to avoid misleading analysis."
            )
    if _wants_percentage(user_query):
        final_sql = _add_pct_column_if_grouped(final_sql)

    # Final SQL cleanup
    final_sql = final_sql.strip().rstrip(";") + ";"
    
    # Validate SQL syntax
    ok, err = validate_sql(final_sql, table_name)
    if not ok:
        return {"sql_ran": False, "sql": final_sql, "sql_result": None, "error": f"Syntax Validation Error: {err}","data_quality_warning": cast_warn,}
    # Schema column precheck
    ok2, err2 = _schema_column_precheck(final_sql, table_name, type_schema)
    if not ok2:
        return {"sql_ran": False, "sql": final_sql, "sql_result": None, "error": f"Schema Validation Error: {err2}","data_quality_warning": cast_warn,}

    try:
        cur = con.execute(final_sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        return {"sql_ran": True, "sql": final_sql, "sql_result": {"columns": cols, "rows": rows, "row_count": len(rows)}, "error": None,"data_quality_warning": cast_warn,}
    except Exception as e:
        return {"sql_ran": False, "sql": final_sql, "sql_result": None, "error": f"Runtime Error: {str(e)}","data_quality_warning": cast_warn,}