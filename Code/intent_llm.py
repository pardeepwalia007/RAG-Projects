
# Interprets user queries into SQL specs using LLM and schema.
import re
from typing import Dict, Any, List, Optional, Tuple
from difflib import get_close_matches

from pydantic import BaseModel, Field, field_validator
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
# ----------------------------
# 1. Generalized Pydantic Model
# ----------------------------
class QuerySpec(BaseModel):
    intent: str = Field(description="The action: 'ranking', 'aggregation','filter', or distribution, 'relationship',visualization ,or metadata ")
    metric: Optional[str] = Field(
        default=None,
        description='The exact numeric column or SQL aggregate (e.g. COUNT(DISTINCT "col")) derived from business rules.'
    )
    # Generalized to List[str] for multi-dimensional analysis
    entity_columns: List[str] = Field(
        default_factory=list, 
        description="List of group-by column names from the schema."
    )
    filter_value: Optional[str] = Field(default=None, description="A specific value, date, year, or condition (e.g. '2023', '> 100') to filter the data.")
    refined_instruction: str = Field(description="A clean technical instruction for a SQL coder")

    # Robust validator to handle LLM string/list confusion
    @field_validator("entity_columns", mode="before")
    @classmethod
    def clean_entities(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v.strip()] if v.strip() else []
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return v

    @field_validator("filter_value", mode="before")
    @classmethod
    def clean_filter(cls, v):
        if isinstance(v, str) and not v.strip():
            return None
        return v


# ----------------------------
# 2. Helpers (Schema & Parsing)
# ----------------------------

def _norm(s: str) -> str:
    """Normalizes strings for comparison."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())

def _extract_quoted_identifiers(expr: str) -> List[str]:
    """Extracts quoted identifiers from expressions."""
    return re.findall(r'"([^"]+)"', expr or "")

def _split_csvish(value: Any) -> List[str]:
    """Splits comma-separated strings into lists."""
    if value is None: return []
    if isinstance(value, list): return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    return [p.strip() for p in s.split(",") if p.strip()] if s else []


# ----------------------------
# 3. Context-Aware Interpreter
# ----------------------------
class QueryInterpreter:
    def __init__(self, con, table_name: str, type_schema: Dict[str, Any]):
        """Initializes interpreter with database connection and schema."""
        self.con = con
        self.table_name = table_name
        self.type_schema = type_schema

        # Dynamic Schema Parsing (Works for ANY CSV)
        self.numeric_cols = _split_csvish(type_schema.get("NUMERIC COLUMNS"))
        self.date_cols = _split_csvish(type_schema.get("DATE COLUMNS"))
        self.text_cols = _split_csvish(type_schema.get("TEXT COLUMNS"))
        
        # self.entity_cols = list(dict.fromkeys(self.text_cols + self.date_cols))
        self.entity_cols = list(dict.fromkeys(self.text_cols + self.date_cols + self.numeric_cols))
        self.pk_col = str(type_schema.get("PRIMARY_KEY_ID") or "").strip() or None

        # Build master list of valid columns
        all_cols = self.numeric_cols + self.entity_cols
        if self.pk_col and self.pk_col not in all_cols:
            all_cols.append(self.pk_col)
        
        self.actual_headers = list(dict.fromkeys(all_cols))
        self.norm_map = {_norm(h): h for h in self.actual_headers}

    # --- Data Sampling (Generalized) ---
    def _escape_sql_like(self, s: str) -> str:
        """Escapes strings for SQL LIKE queries."""
        return (s or "").replace("'", "''")

    def _get_data_context(self, user_query: str) -> Dict[str, List[str]]:
        """
        Samples data values matching query keywords.
        """
        keywords = re.findall(r"\b\w{4,}\b", user_query)
        ignore = {"what", "total", "highest", "lowest", "limit", "trend", "average", "count", "show", "list", "give", "many", "much"}
        keywords = [w for w in keywords if w.lower() not in ignore]

        context: Dict[str, List[str]] = {}
        # Scan first 20 text columns (usually sufficient context)
        for col in self.text_cols[:20]:
            for word in keywords:
                try:
                    safe_word = self._escape_sql_like(word)
                    query = (
                        f'SELECT DISTINCT "{col}" FROM {self.table_name} '
                        f'WHERE CAST("{col}" AS TEXT) ILIKE \'%{safe_word}%\' LIMIT 20'
                    )
                    results = self.con.execute(query).fetchall()
                    if results:
                        context[col] = [str(r[0]) for r in results]
                except Exception:
                    continue
        return context


    def _sanitize_json(self, text: str) -> str:
            """Extracts pure JSON from LLM noise."""
            text = text.strip()
            # Find the first '{' and last '}'
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return text[start : end + 1]
            return text



    # --- Schema Validation (Hard Constraints) ---
    def _coerce_to_schema_column(self, name: str, allowed: List[str]) -> Optional[str]:
        """Maps user column names to exact schema columns."""
        if not name: return None
        if name in allowed: return name
        
        n = _norm(name)
        if n in self.norm_map and self.norm_map[n] in allowed:
            return self.norm_map[n]
        
        # Fuzzy match fallback
        allowed_norm = {_norm(a): a for a in allowed}
        matches = get_close_matches(n, list(allowed_norm.keys()), n=1, cutoff=0.70)
        if matches:
            return allowed_norm[matches[0]]
        return None
    
    def _coerce_metric(self, metric_expr: str) -> str:
        """Validates and corrects metric expressions. If metric is text-but-numeric, use TRY_CAST."""
        m = (metric_expr or "").strip()
        # if not m: raise ValueError("Metric is empty.") below is changed to
        if not m:
            return None
        # if it's an aggregate function, validate inner columns
        if any(x in m.upper() for x in ["COUNT", "SUM", "AVG", "MAX", "MIN"]):
            return m
        # If complex aggregate (SUM, COUNT), validate inner columns
        if "(" in m:
            quoted = _extract_quoted_identifiers(m)
            for q in quoted:
                # Basic PK/ID recovery if user guesses "transaction_id"
                if _norm(q) in ("transactionid", "transaction_id", "id") and self.pk_col:
                     m = m.replace(f'"{q}"', f'"{self.pk_col}"')
                     continue

                fixed = self._coerce_to_schema_column(q, self.actual_headers)
                if not fixed:
                     raise ValueError(f"Invalid column inside metric: {q}")
                m = m.replace(f'"{q}"', f'"{fixed}"')
            return m
            
         # ---- SIMPLE COLUMN METRIC PATH ----

        # 1) If it's already a known numeric column, accept
        fixed_num = self._coerce_to_schema_column(m, self.numeric_cols)
        if fixed_num:
            return fixed_num

        # 2) If it's a text column, try numeric sniff + cast
        fixed_text = self._coerce_to_schema_column(m, self.text_cols)
        if fixed_text:
            try:
                # Clean common junk like commas, currency symbols, pipes, spaces, etc.
                cleaned = f"regexp_replace(\"{fixed_text}\", '[^0-9\\.\\-]+', '', 'g')"

                q_cast = f"""
                    SELECT
                        COUNT(*) AS total_rows,
                        SUM(CASE WHEN TRY_CAST({cleaned} AS DOUBLE) IS NOT NULL THEN 1 ELSE 0 END) AS castable_rows
                    FROM {self.table_name};
                """
                total_rows, castable_rows = self.con.execute(q_cast).fetchone()
                total_rows = int(total_rows or 0)
                castable_rows = int(castable_rows or 0)

                if total_rows > 0 and (castable_rows / total_rows) >= 0.90:
                    # Return the same cleaned+casted expression for downstream SQL generation
                    return f"TRY_CAST({cleaned} AS DOUBLE)"
            except Exception:
                pass

        # 3) Otherwise reject
        raise ValueError(f"Metric '{m}' is not a numeric column.")
    # --- Heuristic Label Column Picker ---
    def _pick_label_column(self) -> Optional[str]:
        if not self.text_cols:
            return None

        bad_patterns = ("id", "link", "url", "image", "img", "review", "content", "about", "description")
        good_patterns = ("name", "title", "product", "item", "category", "brand", "model", "type")

        candidates = []
        for c in self.text_cols:
            lc = c.lower()

            if any(p in lc for p in bad_patterns):
                continue
            if self.pk_col and c == self.pk_col:
                continue

            score = 0
            if any(p in lc for p in good_patterns):
                score += 3

            try:
                total, distinct = self.con.execute(
                    f'SELECT COUNT(*) AS t, COUNT(DISTINCT "{c}") AS d FROM {self.table_name};'
                ).fetchone()
                total = int(total or 0)
                distinct = int(distinct or 0)
                if total > 0:
                    uniq_ratio = distinct / total
                    if 0.01 <= uniq_ratio <= 0.80:
                        score += 2
                    elif uniq_ratio < 0.01:
                        score -= 1
                    else:
                        score -= 2
            except Exception:
                pass

            try:
                avg_len = self.con.execute(
                    f'SELECT AVG(LENGTH(CAST("{c}" AS VARCHAR))) FROM {self.table_name};'
                ).fetchone()[0]
                if avg_len is not None and float(avg_len) <= 80:
                    score += 1
                elif avg_len is not None and float(avg_len) >= 200:
                    score -= 2
            except Exception:
                pass

            candidates.append((score, c))

        if not candidates:
            return self.text_cols[0]

        candidates.sort(reverse=True, key=lambda x: x[0])
        return candidates[0][1]


    # --- Spec Normalization (Soft Heuristics) ---
    def _normalize_spec(self, spec: QuerySpec) -> QuerySpec:
        intent = (spec.intent or "").strip().lower()

        if intent == "distribution" and not spec.metric and spec.entity_columns:
            spec.metric = spec.entity_columns[0]
            spec.entity_columns = []

        if intent == "distribution":
            candidate = (spec.metric or "").strip()
            # distribution of a text column => frequency distribution
            if candidate and candidate in self.text_cols:
                spec.intent = "ranking"
                spec.entity_columns = [candidate]
                spec.metric = "COUNT(*)"
                spec.refined_instruction = f'Count records by "{candidate}" and sort descending.'
                intent = "ranking" # update intent for downstream logic

        #  dataset-agnostic ranking label pick
        if intent == "ranking" and not spec.entity_columns:
            label = self._pick_label_column()
            if label:
                spec.entity_columns = [label]

        if intent == "ranking":
            q = (spec.refined_instruction or "").lower()

            plural_entity = any(
                e.lower().endswith("s") for e in spec.entity_columns
            )

            superlative = re.search(
                r"\b(highest|lowest|most|least|best|worst)\b", q
            ) is not None

            explicit_single = re.search(
                r"\b(single|one|only|top\s*1|highest\s*one)\b", q
            ) is not None

            if plural_entity and superlative and not explicit_single:
                # Signal downstream layers NOT to collapse to LIMIT 1
                spec.refined_instruction += " Rank multiple entities (top-N comparison, not single result)."

        if intent == "aggregation" and not spec.metric:
            spec.metric = 'COUNT(*)'

        if intent == "filter" and not spec.entity_columns and self.text_cols:
            spec.entity_columns = [self.text_cols[0]]

        return spec

    # --- Hard Enforcement (Schema Grounding) ---
    def _hard_enforce(self, spec: QuerySpec) -> QuerySpec:
        """
        Enforces schema compliance on query specs.
        """
        spec = self._normalize_spec(spec)
        if spec.metric: # based on coerce_metric changes if metric is not None
            spec.metric = self._coerce_metric(spec.metric)

        # Handle entities (plural)
        validated_entities = []
        for col in spec.entity_columns:
            fixed = self._coerce_to_schema_column(col, self.entity_cols)
            if fixed:
                validated_entities.append(fixed)
        
        # If user asked for aggregation but provided no valid group-by, default to PK or nothing
        if not validated_entities and spec.intent == 'aggregation':
             # Optional: Default to PK if nothing else found, or leave empty for scalar
             pass 

        spec.entity_columns = list(dict.fromkeys(validated_entities))
        return spec

    # ----------------------------
    # 4. Main Intelligence (Prompt-Driven)
    # ----------------------------
    def refine_intent(self, user_query: str, business_context: str = None,retry_context: str = None) -> QuerySpec:
        """
        Refines user query into validated QuerySpec using LLM.
        """
        parser = PydanticOutputParser(pydantic_object=QuerySpec)
        llm = ChatOllama(model="qwen2.5:14b-instruct-q5_K_M", format="json", temperature=0)  # prev qwen2.5:14b changed from 7b instruct
    
        data_context = self._get_data_context(user_query)

        # Generalized System Prompt
        system_template = """You are a Data Architect. Map the user query to the schema using the provided Business Context.

        BUSINESS DEFINITIONS & RULES (Priority):
        {business_context}

        DATABASE SCHEMA:
        - Numeric Columns (Metrics): {metrics}
        - Text/Date Columns (Entities): {entities}
        - Primary Key: {primary_key}
        - Valid Data Samples: {data_context}

        INSTRUCTIONS:
        1. INTENT CLASSIFICATION:
           - "List all columns", "Show fields", "What data do you have?": Set intent to 'metadata'.
           - "Distribution of X": Set intent to 'distribution'.
           - "How does X relate to Y?", "Correlation between X and Y", "Scatter plot X vs Y": 'relationship'.
           -  otherwise use 'ranking', 'aggregation', or 'filter'.
        2. METRIC SELECTION:
           - For 'relationship', you usually need TWO metrics (e.g., Shipping Cost vs Distance).
           - Put the Dependent Variable (Y-axis) in 'metric'.
           - Put the Independent Variable (X-axis) in 'entity_columns'.

        3. ENTITY SELECTION: Return a LIST of columns to group by.

        4. 
         REFINED INSTRUCTION (THE MOST IMPORTANT PART):
           - You must rewrite the user's question into "SQL-Speak".
           - **SIMPLIFY**: Break down complex questions.
           - **ENUMERATE (CRITICAL)**: If the user asks for "attributes", "factors", "drivers", "features", or "all columns", YOU MUST SELECT 3-5 RELEVANT COLUMNS from the provided schema.
             - Bad: "Analyze all attributes." (SQL Agent cannot handle vague requests)
             - Good: "Group by [Category_Column] and calculate averages for [Metric_A], [Metric_B], and [Metric_C]."
             - COMPARE: If comparing groups, REMOVE 'LIMIT 1'. You need all groups to compare.
             - Bad: "Find the most frequent..." (Triggers LIMIT 1 which hides data)
             - Good: "Group by [Category_Column] and count records to compare distribution across all groups."
         FILTERING LOGIC (Critical):
           - **FILTER**: If the user implies a filter, include specific conditions (e.g. "Filter where Status = 'Active'").
            - TEXT COLUMNS: If the user search term (e.g. "Macbook") is a SUBSTRING of real values, specify "using partial match".
            - DATE/TIME COLUMNS (STRICT): 
                - NEVER suggest "partial match" for dates.
                - If the user mentions a year (e.g. '2023'), extract '2023' as the filter_value.
                - The instruction should be: "Filter by Year 2023 using native date functions."

        5. DO NOT answer the user. ONLY return JSON that matches the QuerySpec schema exactly. No extra keys.
        {format_instructions}
        """
    # --- YOUR NEW RETRY/AUDIT TEMPLATE ---
        audit_template = """CRITICAL: PREVIOUS SQL ATTEMPT FAILED. Switch to STRICT AUDIT MODE.
    
        ERROR LOG & SCHEMA FEEDBACK:
        {retry_context}

        BUSINESS RULES TO RE-EXAMINE:
        {business_context}

        DATABASE SCHEMA:
        - Numeric Columns (Metrics): {metrics}
        - Text/Date Columns (Entities): {entities}

        STRICT INSTRUCTIONS:
        1. DO NOT use any column name that is not in the Numeric or Text lists above.
        2. If the previous error was 'Unknown column', check if you hallucinated a column (like 'quantity').
        3. INTENT CORRECTION:
           - If the error mentions "Aggregates cannot be nested" or massive grouping failures, change intent to 'metadata' or 'filter' and simplify the query to 'SELECT * LIMIT 5'.
        4. RE-READ the Business Rules for the correct formula:
           - If 'Sales Volume' or 'Units' is requested and 'quantity' is missing, you MUST use 'COUNT(*)'.
           - DO NOT use 'SUM(money)' for Volume; that is for Revenue.
        5. DO NOT answer the user. ONLY return JSON that matches the QuerySpec schema exactly. No extra keys.
        {format_instructions}
        """
        # Logic to switch templates based on the presence of retry_context
        chosen_template = audit_template if retry_context else system_template

        prompt = ChatPromptTemplate.from_messages([
            ("system", chosen_template),
            ("human", "{query}")
        ])

        chain = prompt | llm 
        
        # Invoke with retrieval context
        spec = chain.invoke({
            "metrics": self.numeric_cols,
            "entities": self.entity_cols,
            "primary_key": self.pk_col or "id",
            "data_context": data_context,
            "business_context": business_context or "No rules provided.",
            "retry_context": retry_context or "No previous errors.", # ADD THIS LINE
            "query": user_query,
            "format_instructions": parser.get_format_instructions()
        })
        # Manual Sanitization Step
        clean_json = self._sanitize_json(spec.content)
        
        try:
            spec = parser.parse(clean_json)
        except Exception:
            # Fallback: Try OutputFixingParser if raw parse fails
            fix_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
            spec = fix_parser.parse(clean_json)

        # Final structural validation (Schema Grounding)
        return self._hard_enforce(spec)
