import re
from typing import Dict, Any
# Determines if a question requires SQL execution for data analysis.
def should_run_sql(questions: str, schema: Dict[str, Any] = None) -> bool:
    """
    Decides whether to run SQL based on question intent AND dynamic schema columns.
    """
    q = questions.lower()

    # 1. HARD BLOCKERS (Definitions/Rules) -> Vector DB
    if any(re.search(rf"\b{word}\b", q) for word in ["meaning", "definition", "formula", "prohibited", "policy"]):
        return False

    # ---------------------------------------------------------
    # DYNAMIC KEYWORD EXTRACTION (The "Automatic" Part)
    # ---------------------------------------------------------
    dynamic_tokens = []
    if schema:
        # Combine Numeric and Text column strings
        raw_cols = (
            str(schema.get("NUMERIC COLUMNS", "")) + "," + 
            str(schema.get("TEXT COLUMNS", ""))
        )
        # Clean and split: "Heart_Disease" -> ["heart", "disease", "heart disease"]
        for col in raw_cols.split(","):
            clean_col = col.strip().lower()
            if not clean_col: continue
            
            # Add the exact column name (normalized)
            normalized = clean_col.replace("_", " ")
            dynamic_tokens.append(normalized)
            
            # Add individual words if snake_case (e.g., "cholesterol" from "serum_cholesterol")
            if "_" in clean_col:
                dynamic_tokens.extend(clean_col.split("_"))

    # ---------------------------------------------------------
    # STATIC KEYWORDS (Foundational Logic)
    # ---------------------------------------------------------
    # 2. METRIC TOKENS (What are we measuring?)

        metrics = [
            # Visuals (ADD THESE)
            "plot", "graph", "chart", "visualize", "show me", "diagram",
            # Sales/Retail
            "revenue", "sales", "sold", "units", "quantity", "orders", 
            "aov", "price", "data", "earned", "bought", "transactions", 
            "money", "cash", "income", "profit", "amount", "value",
            # HR/Employees
            "employees", "employee", "employe", "staff", "workers", "headcount", 
            "salary", "salaries", "compensation", "pay",
            "age", "tenure", "hired", "fired", "terminated", "left",
            "name", "names", "full name", "person", "people",
            # Healthcare (NEW)
            "patient", "patients", "doctor", "doctors", "hospital", "hospitals",
            "condition", "diagnosis", "blood", "insurance", "billing", "bill",
            "medication", "medicine", "drug", "prescribed", "prescription",
            "test", "result", "results", "room", "stay",
            # General Data Tokens
            "numeric", "numerical", "columns", "column", "records", "rows", "dataset", "table",
            "cost", "costs", "shipping", "shipment", "freight", "weight", 
            "distance", "transit", "delivery", "deliveries", "delay", "delays",
            "money", "cash", "income", "profit", "amount", "value", "valuation",
            "company", "companies", "business", "startup", "startups", "unicorn", "unicorns",
            # Visuals
                "plot", "graph", "chart", "visualize", "show me", "diagram", "scatter",
                # Data Tokens
                "numeric", "numerical", "columns", "column", "records", "rows", "dataset", "table", "data",
                # Business/Generic
                "total", "sum", "average", "avg", "count", "amount", "value", "quantity", 
                "number", "numbers", "figure", "figures", # [ADDED MISSING KEYWORDS]
                "rate", "percentage", "ratio", "score", "level", "levels", "range"
        ]
        
        # 3. STAT & INTENT TOKENS (How are we asking?)
        rankings_and_stats = [
            # Aggregations
            "total", "sum", "average", "avg", "count", "number of", "how many",
            "how much",
            # Ranking/Sorting
            "most", "least", "best", "worst", "highest", "lowest", "top", "bottom",
            # Analysis & Description
            "unique", "distinct", "trend", "spread", "distribution", "breakdown",
            "descriptive", "describe", "description", "analysis", "analyze", 
            "stats", "statistics", "summary", "profile",
            # Listing
            "list", "show", "give me", "find", "who", "which", "what is", "what are",
            "what", "present",
            "available", "exist", "existing", "used",
            "most", "least", "best", "worst", "highest", "lowest", "top", "bottom",
            "unique", "distinct", "trend", "distribution", "breakdown", "analysis", 
            "summary", "profile", "list", "compare", "comparison", "difference", "differ",
            "relate", "relationship", "correlation", "impact", "affect", "between",
            "vs", "versus",
            "change", "changed", "growth", "growing", "evolution", "vary", "varies" # [ADDED MISSING KEYWORDS]
        ]
        
        # 4. DIMENSIONAL TOKENS (Grouping categories)
        dimensions = [
            # Retail
            "product", "category", "store", "brand", "item", "coffee", "drink",
            # HR
            "department", "division", "role", "title", "position", "gender", 
            "location", "state", "city", "manager", "supervisor",
            "contract", "full time", "part time", "permanent", "temporary", "intern", "active",
            # Healthcare (NEW)
            "admission", "admitted", "discharge", "emergency", "urgent", "elective", 
            "type", "provider", "medical",
            # Time/General
            "month", "year", "date", "per", "by", "daily", "monthly",
            "time", "weekly", "quarterly", "period",
            "category", "sector", "industry", "industries", "domain", "field", 
            "city", "country", "continent", "region", "location","categories"
        ]

    # ---------------------------------------------------------
    # MERGE & CHECK
    # ---------------------------------------------------------

    # [CRITICAL] Add dynamic schema tokens to the search list
    # If the user asks about "Cholesterol" and it's in the CSV, we catch it here.
    combined_vocab = set(metrics + dimensions + dynamic_tokens)

    # LOGIC CHECK
    has_vocab_match = any(token in q for token in combined_vocab)
    has_rank_stat = any(s in q for s in rankings_and_stats)
    
    # Rule A: The question mentions a known column/metric AND a statistical intent
    # e.g., "How does [age] [differ]..."
    if has_vocab_match and has_rank_stat:
        return True

    # Rule B: Direct "Show me" or "Plot" intent with any column
    if any(v in q for v in ["plot", "graph", "chart", "list", "show", "what is", "what are"]) and has_vocab_match:
        return True

    # Rule C: Comparison logic ("between patients")
    if "between" in q and has_vocab_match:
        return True

    return False
# Validates SQL queries for safety and correctness.
def validate_sql(sql_query: str, table_name: str) :
    """Ensures SQL queries are safe SELECT statements referencing the correct table."""
    if not sql_query:
        return False, "No SQL generated."
        
    # Convert both to upper for a fair comparison
    clean_sql = sql_query.strip().upper()
    target_table = table_name.strip().upper()
    
    # 1. Start check
    if not clean_sql.startswith("SELECT"):
        return False, "Only SELECT statements are permitted."

    # 2. Dangerous keyword check
    forbidden = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE"]
    if any(re.search(rf"\b{word}\b", clean_sql) for word in forbidden):
        return False, "Dangerous keyword detected."

    # 3. Table Check (FIXED: Case-insensitive and robust)
    if target_table not in clean_sql:
        return False, f"Query must reference the table: {table_name}"

    return True, None



# Checks if SQL query returns a single aggregate value.
def is_scalar_intent(sql_query: str) -> bool:
    """Identifies if query is a global aggregate returning one row."""
    q = sql_query.upper()
    aggregates = ["SUM(", "COUNT(", "AVG(", "MIN(", "MAX("]
    return any(a in q for a in aggregates) and "GROUP BY" not in q


# limiting the query result
# Enforces result limits to prevent excessive data retrieval.
def enforce_safety_limits(
    question: str,
    sql_query: str,
    max_limit: int = 1000,
    intent: str | None = None
) -> str:
    """Applies appropriate LIMIT clauses to SQL queries based on question intent."""
    sql = (sql_query or "").strip().rstrip(";")
    q = (question or "").lower()

    # 1) Identify scalar (single-value) queries and strip LIMIT
    # (e.g., SELECT COUNT(*)...)
    if is_scalar_intent(sql):
        sql = re.sub(r"\bLIMIT\s+\d+\b", "", sql, flags=re.IGNORECASE).strip()
        return f"{sql};"

    # 2) If question explicitly asks for N (e.g. "Top 10", "First 5")
    m = re.search(r"\b(top|first|limit|show)\s*(\d+)\b", q, re.IGNORECASE)
    requested_limit = int(m.group(2)) if m else None

    # 3) Check if user specifically asked for "ALL" or "LIST"
    # This overrides the "ranking" default of 1
    wants_all = re.search(r"\b(all|list|distinct|show)\b", q, re.IGNORECASE) is not None

    # 4) Decide final limit
    if requested_limit:
        limit = min(requested_limit, max_limit)
    
    # [CRITICAL FIX] If intent is Trending/Distribution, NEVER default to 1.
    # We need high volume data for charts, even if the user says "Most".
    elif intent in ["relationship", "distribution", "visualization"]:
        limit = max_limit
        
    # [FIX] If "ranking" intent, ONLY default to 1 if user didn't ask for "all"
    elif intent == "ranking":
        if wants_all:
             limit = max_limit # "Rank all doctors" -> Show all
        else:
             limit = 5 # "Rank doctors" -> Show top 5 (Better default than 1)
    
    else:
        # Fallback logic
        wants_single = re.search(r"\b(highest|lowest|most|least|best|worst)\b", q) is not None
        
        # [FIX] Even if "wants_single" is True, if they said "List all", "All" wins.
        if wants_single and not wants_all:
            limit = 1
        else:
            limit = max_limit

    # 5) Remove any existing LIMIT and enforce the new one
    sql = re.sub(r"\bLIMIT\s+\d+\b", "", sql, flags=re.IGNORECASE).strip()
    return f"{sql} LIMIT {limit};"

UNIQUE_HINTS = r"\b(unique|distinct|dedup|de-dup|non-duplicate)\b"

def enforce_transaction_semantics(question: str, sql: str, pk: str) -> str:
    """Adjusts SQL for correct unique vs. total transaction counting based on question."""
    q = question.lower()

    asked_unique = re.search(UNIQUE_HINTS, q) is not None
    asks_compare = (
        re.search(r"\b(vs|versus|compare|comparison|both)\b", q) is not None
        and asked_unique
    )

    # If NOT unique requested, strip DISTINCT for pk counts
    if not asked_unique:
        sql = re.sub(
            rf"COUNT\s*\(\s*DISTINCT\s+{re.escape(pk)}\s*\)",
            f"COUNT({pk})",
            sql,
            flags=re.IGNORECASE
        )
        return sql

    # If compare requested, ensure "total" aliases are NOT distinct
    if asks_compare:
        sql = re.sub(
            rf"COUNT\s*\(\s*DISTINCT\s+{re.escape(pk)}\s*\)\s+AS\s+(total_transactions|transaction_count|total|transactions)\b",
            rf"COUNT({pk}) AS \1",
            sql,
            flags=re.IGNORECASE
        )
        return sql

    return sql



TOTAL_HINTS = r"\b(total|overall|sum|all\s*time)\b"
REVENUE_HINTS = r"\b(revenue|sales)\b"

# Adjusts revenue aggregation for total calculations in grouped queries.
def enforce_revenue_semantics(question: str, sql: str, revenue_col: str = "revenue") -> str:
    """Changes MAX to SUM for revenue in grouped queries when total is requested."""
    q = (question or "").lower()
    s = (sql or "").strip()

    wants_total = re.search(TOTAL_HINTS, q) is not None
    mentions_revenue = re.search(REVENUE_HINTS, q) is not None

    if not (wants_total and mentions_revenue):
        return s

    # Only if query is grouped (i.e., per product/category)
    if "GROUP BY" not in s.upper():
        return s

    # Rewrite MAX(revenue) -> SUM(revenue)
    s = re.sub(
        rf"MAX\s*\(\s*\"?{re.escape(revenue_col)}\"?\s*\)",
        f'SUM("{revenue_col}")',
        s,
        flags=re.IGNORECASE
    )

    # Rename alias if needed
    s = re.sub(r"\bmax_revenue\b", "total_revenue", s, flags=re.IGNORECASE)
    return s


RANK_HINTS = r"\b(top|highest|most|best|lowest|least|bottom)\b"


# Ensures ranking queries have proper ORDER BY and LIMIT.
def enforce_ranking_shape(
    question: str,
    sql: str,
    default_limit: int = 10,
    intent: str | None = None
) -> str:
    """Adds ORDER BY and LIMIT to ranking queries for correct top/bottom results."""
    q = (question or "").lower()
    s = (sql or "").strip().rstrip(";")

    # ranking if either:
    # - interpreter said intent='ranking'
    # - or question contains ranking hints
    is_rank = (intent == "ranking") or (re.search(RANK_HINTS, q) is not None)
    if not is_rank:
        return f"{s};"

    # Add ORDER BY if missing (best effort: last alias)
    if re.search(r"\bORDER\s+BY\b", s, flags=re.IGNORECASE) is None:
        aliases = re.findall(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)\b", s, flags=re.IGNORECASE)
        if aliases:
            metric_alias = aliases[-1]
            s = f"{s} ORDER BY {metric_alias} DESC"
        else:
            s = f"{s} ORDER BY 2 DESC"  # Fallback to first column
    # Ensure LIMIT exists
    if re.search(r"\bLIMIT\s+\d+\b", s, flags=re.IGNORECASE) is None:
        s = f"{s} LIMIT {default_limit}"
    
    
    return f"{s};"

def enforce_distribution_shape(
    final_sql: str,
    table_name: str,
    metric_col: str,
    limit: int = 1000
) -> str:
    """
    If intent is distribution, force SQL to return raw metric values (no GROUP BY, no SUM/COUNT).
    Keeps the WHERE clause if present.

    Supports metric_col being either:
    - a simple column name (e.g. rating_count)
    - an expression (e.g. TRY_CAST("rating_count" AS DOUBLE))
    """

    sql_l = final_sql.lower()

    has_group_by = "group by" in sql_l
    has_agg = re.search(r"\b(sum|avg|count|min|max)\s*\(", sql_l) is not None

    # if not (has_group_by or has_agg):
    #     return final_sql

    # Extract WHERE clause if it exists
    where_clause = ""
    m = re.search(
    r"\bwhere\b(.+?)(\bgroup\s+by\b|\border\s+by\b|\blimit\b|;|\)\s*select\b|$)",
    final_sql,
    flags=re.IGNORECASE | re.DOTALL
    )
    if m:
        where_clause = " WHERE " + m.group(1).strip()

    metric_expr = (metric_col or "").strip()

    # If it's an expression (TRY_CAST(...), CAST(...), or anything with parentheses), don't quote it
    is_expr = bool(re.search(r"\b(try_cast|cast)\s*\(", metric_expr, flags=re.IGNORECASE)) or "(" in metric_expr

    select_metric = metric_expr if is_expr else f'"{metric_expr}"'

    return f"SELECT {select_metric} AS value FROM {table_name}{where_clause} LIMIT {limit};"