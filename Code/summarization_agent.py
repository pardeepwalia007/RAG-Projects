import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Extracts risks from business rules by filtering markdown noise.
def extract_risks(rules: str) -> str:
    """Filters business rules to extract risk-related content, removing markdown headers."""
    kept = []
    for line in (rules or "").splitlines():
        s = line.strip()
        if not s:
            continue
        # Skip only "pure" markdown headers, but KEEP numbered rule headers like "## 2. ..."
        if s.startswith("#") and not re.match(r"^#+\s*\d+[\.\)]", s):
            continue
        kept.append(s)
    return "\n".join(kept) if kept else "- No specific business risks identified."

def _safe(x):
    if x is None:
        return ""
    # Format floats cleanly (general reporting)
    if isinstance(x, float):
        return f"{x:.2f}"
    s = str(x)
    # Escape pipe characters for markdown tables
    s = s.replace("|", "\\|")
    s = s.replace("\n", " ").replace("\r", " ")
    return s
def _clean_where_for_humans(where_clause: str) -> str:
    """
    Removes TRY_CAST/regexp_replace noise from WHERE clause so the summarizer
    doesn't copy technical casting into Method.
    Keeps the business meaning: e.g. TRY_CAST(regexp_replace("x"...)) > 50  ->  "x" > 50
    """
    if not where_clause:
        return ""

    s = where_clause

    # Replace TRY_CAST(regexp_replace("col", ... ) AS DOUBLE) with "col"
    # (handles whitespace/newlines via DOTALL)
    s = re.sub(
        r"TRY_CAST\s*\(\s*regexp_replace\(\s*\"([^\"]+)\".*?\)\s*AS\s*DOUBLE\s*\)",
        r'"\1"',
        where_clause,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Replace TRY_CAST("col" AS DOUBLE) with "col"
    s = re.sub(
        r"""TRY_CAST\s*\(\s*"([^"]+)"\s*AS\s+DOUBLE\s*\)""",
        r'"\1"',
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_real_cast_alert(alert: str, sql_text: str) -> bool:
    a = (alert or "").lower()
    s = (sql_text or "").lower()
    return ("try_cast" in a) or ("numeric type conversion" in a) or ("try_cast" in s)




# Formats SQL result rows into narrative observations for LLM processing.
def format_rows(headers, rows, n=50): 
    """Formats SQL result rows into narrative observations, handling large datasets with truncation."""
    if not rows:
        return "STATUS: NO_QUANTITATIVE_DATA_FOUND"
    
    total = len(rows)
    truncated = total > n

    out = []
    out.append(f"METADATA_RESULT_COUNT: {total}")
    out.append(f"RESULT_TRUNCATED: {str(truncated).upper()}") 
    
    if truncated and n >= 4:
        k = n // 2
        head = rows[:k]
        tail = rows[-(n - k):]
        
        # 1. Process Head
        for i, row in enumerate(head):
            vals = ", ".join([f"{_safe(h)}={_safe(r)}" for h, r in zip(headers, row)])
            out.append(f"Row {i+1}: {vals}")
            
        # 2. INSERT EXPLICIT SEPARATOR
        skipped_count = total - len(head) - len(tail)
        out.append(f"\n... [SKIPPING {skipped_count} ROWS OF DATA FOR READABILITY] ...\n")

        # 3. Process Tail
        for i, row in enumerate(tail):
            # Calculate true index for accuracy
            true_idx = total - len(tail) + i + 1
            vals = ", ".join([f"{_safe(h)}={_safe(r)}" for h, r in zip(headers, row)])
            out.append(f"Row {true_idx}: {vals}")

    else:
        for i, row in enumerate(rows[:n]):
            vals = ", ".join([f"{_safe(h)}={_safe(r)}" for h, r in zip(headers, row)])
            out.append(f"Row {i+1}: {vals}")

    return "\n".join(out)



def summarize_with_llama(question: str, evidence: dict, source_type: str,):
    """Generates a constrained LLM summary of query results, ensuring factual accuracy and professional formatting."""
    llm = ChatOllama(
        model="qwen2.5:14b-instruct-q5_K_M", 
        temperature=0,
        stop=["```sql", "USER QUESTION:", "Pipeline Finished", "Note:", "Fact Check:"]
    )

    sql_out = evidence.get("sql_output") or {}
    

    # Robust extraction
    if isinstance(sql_out, dict) and sql_out.get("sql_result"):
        sql_res = sql_out["sql_result"]
        headers = sql_res.get("columns") or []
        rows = sql_res.get("rows") or []
    else:
        headers = []
        rows = []

    sql_query = (sql_out.get("sql") or "") if isinstance(sql_out, dict) else ""

    # Check for Visualization in the Evidence
    has_viz = bool(evidence.get("viz_data"))
    viz_status_line = f"VIZ_STATUS: {'GENERATED' if has_viz else 'NONE'}"
    viz_method = evidence.get("viz_method")
    viz_method_line = f"VIZ_METHOD: {viz_method}" if viz_method else "VIZ_METHOD: None"
    m = re.search(
        r"\bWHERE\b(.*?)(\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|$)",
        sql_query,
        flags=re.IGNORECASE | re.DOTALL
    )
    where_clause = m.group(1).strip() if m else ""
    where_clause_clean = _clean_where_for_humans(where_clause)

    filter_block = f"OBSERVED_FILTERS: {where_clause_clean}" if where_clause_clean else "OBSERVED_FILTERS: None"

    quality_alert = evidence.get("data_quality_warning")
    sql_text = sql_query or ""

    if quality_alert and not _is_real_cast_alert(quality_alert, sql_text):
        quality_alert = None

    quality_block = f"DATA_QUALITY_ALERT: {quality_alert}" if quality_alert else "DATA_QUALITY_ALERT: None"

    intent_val = evidence.get("intent") or "unknown"
    intent_line = f"INTENT: {intent_val}"

    # Build a single, authoritative method line the model must copy verbatim
    intent_val = (evidence.get("intent") or "unknown").strip().lower()

    if has_viz and viz_method:
        method_line = f"METHOD_LINE: {viz_method}"
    else:
        # For non-viz cases, method should come from intent + observed filters (cleaned)
        # This prevents vague “based on filter criteria” hallucinations.
        if where_clause_clean:
            method_line = f"METHOD_LINE: {intent_val} using observed filters: {where_clause_clean}"
        else:
            method_line = f"METHOD_LINE: {intent_val} with no observed filters"
    
    data_block = format_rows(headers, rows)

    is_truncated = "RESULT_TRUNCATED: TRUE" in data_block
    if has_viz:
        notes_line = "NOTES_LINE: Text output is truncated for readability, but the visualized chart includes the full dataset." if is_truncated \
            else "NOTES_LINE: See chart for detailed distribution."
    else:
        notes_line = "NOTES_LINE: Text output is truncated for readability; aggregation is calculated over all matching records." if is_truncated \
            else "NOTES_LINE: Aggregation is calculated over all matching records."


    data_block = (
    viz_status_line
    + "\n"
    + viz_method_line+ "\n"
    + method_line
    + "\n"
    + filter_block
    + "\n"
    + quality_block
    + "\n"
    + intent_line
    + "\n"+ notes_line
    + "\n"
    + data_block
    )   

    rules_full = evidence.get("business_rules", "N/A")
    risks_only = extract_risks(rules_full)

    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Senior BI Analyst.

STRICT OPERATIONAL CONSTRAINTS:
1) NEVER mention technical database terms (SQL, Query, Table, Database).
     
2) NEVER invent, explain, infer, or perform calculations, formulas, ratios, or averages.
     
3) ONLY restate facts that appear verbatim in <OBSERVED_DATA>.
     
4) If <OBSERVED_DATA> contains "NO_QUANTITATIVE_DATA_FOUND":
   - Do NOT say "No data found."
   - INSTEAD, say: "I could not find specific data matching that request."
   - THEN, suggest a retry step (e.g., "You may try rephrasing your question or checking the data source for relevant information.")
     
5) Use ONLY verbatim citations from <RISKS_ALLOWED> for the RISKS / LIMITATIONS section.
     
6) Start immediately with 'EXECUTIVE SUMMARY'.
     
7) Do NOT introduce rankings unless explicit.
     
8) IF PRESENT ALWAYS format currency with an escaped dollar sign (e.g., \$287,139.00) or use "USD". If the metric is a count, score, or rating, do NOT use USD; use the appropriate unit (e.g., '14.5% success rate' or 'Rating Score').
     
9) For TREND/TIME-SERIES analysis:
   - Do NOT use emotive language ("surged", "plummeted", "drastic change").
   - Do NOT infer causality.
   - Simply state the **Range** (Min to Max) and specific **Peak** values verbatim.
     
10) DATA QUALITY RULE: If DATA_QUALITY_ALERT contains a warning, you must include a "Data Quality" bullet point under the RISKS / LIMITATIONS section. Explicitly state that certain non-numeric values were excluded and include the phrase: "To ensure a more accurate analysis, it is recommended to clean the source data."
     10.1) DATA QUALITY PLACEMENT RULE: If DATA_QUALITY_ALERT is not None, it must appear ONLY under RISKS / LIMITATIONS as a bullet. Never put it in Notes.     

11) If INTENT == 'distribution', NEVER use time-based language (e.g., monthly, daily, trend, over time).   

[RESPONSE FORMAT GUIDELINES]

A. **For Lists & Rankings** (e.g., "Top 5", "List of carriers"):
   - Use a **Markdown Table**.
   - Example:
     | Carrier | Total Weight |
     |---------|--------------|
     | FedEx   | 13,651.00 kg |

B. Trends with Visualization (ONLY when VIZ_STATUS == 'GENERATED' AND (INTENT == 'trend' OR a date/time field is present)):
   - Use the **Executive Block** format to summarize the chart:
     * **Entity**: [Trend Name]
     * **Metric**: [Value Range, e.g., "/\$29,824.57 to /\$44,472.30"]
     * **Method**: copy VIZ_METHOD verbatim from <OBSERVED_DATA>. If VIZ_METHOD is None, write: "Method: Not available."
     * **Notes**: "See chart for detailed trend."

B2.  Distributions with Visualization (ONLY when VIZ_STATUS == 'GENERATED' AND INTENT == 'distribution'):
   - Use the Executive Block format:
     * **Entity**: [Distribution Name]
     * **Metric**: [Observed value range, e.g., "219 to 32,999"]
     * **Method**:  copy VIZ_METHOD verbatim from <OBSERVED_DATA>. If VIZ_METHOD is None, write: "Method: Not available."
     * **Notes**: "See chart for detailed distribution."

C. **For Single Metrics** (e.g., "Total Revenue"):
   - Use the **Executive Block** format:
     * **Entity**: [Name]
     * **Metric**: [Value]
     * **Method**: copy METHOD_LINE verbatim from <OBSERVED_DATA>. If METHOD_LINE is missing, write: "Method: Not available."
     * **Notes**: 
         - If VIZ_STATUS is NONE: do NOT mention charts (no "See chart..."). 
         - Otherwise, follow the visualization notes rules.

D. **Truncation & Visualization Disclaimer**:
   - If "VIZ_STATUS: GENERATED":*Text output is truncated for readability, but the visualized chart includes the full dataset."*
   - If "VIZ_STATUS: NONE":*Text output is truncated for readability; aggregation is calculated over all matching records."*

E. **Empty Data Fallback**:
   - Focus entirely on the "Retry Step" and obtaining user permission.

F. If VIZ_STATUS == 'NONE':
- NEVER write "Method: Not available"
-Notes: copy NOTES_LINE verbatim from <OBSERVED_DATA>
[REQUIRED RESPONSE FORMAT]
(Dynamic: Use Table for lists, Block for singletons/trends. Always check Truncation/Viz status.)
"""),
    ("user", """
<BUSINESS_DEFINITIONS>
{rules}
</BUSINESS_DEFINITIONS>

<RISKS_ALLOWED>
{risks}
</RISKS_ALLOWED>

<OBSERVED_DATA>
{data}
</OBSERVED_DATA>

TASK:
Answer "{question}" using ONLY the information provided above.
""")
])

    text = (prompt | llm).invoke({
        "question": question,
        "rules": rules_full,
        "risks": risks_only,
        "data": data_block
    }).content.strip()

    critical_bad = ["[insert", "tbd", "to be determined", "[insert number]"]
    if any(p in text.lower() for p in critical_bad):
        return "EXECUTIVE SUMMARY:\nValidation Error.\n\nKEY FINDINGS:\n• No reliable data extracted.\n\nRISKS / LIMITATIONS:\n- System safety fallback."

    return text