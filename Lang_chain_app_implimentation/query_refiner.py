# Refines user queries into schema-compliant SQL specs.
import re
from typing import Dict, Any, List, Optional, Tuple
from difflib import get_close_matches

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
 # Pydantic v2 style
from pydantic import field_validator

class QuerySpec(BaseModel):
    intent: str = Field(description="The action: 'ranking', 'aggregation', or 'filter'")
    metric: str = Field(
        description='Either a numeric column name from schema OR an aggregate like COUNT(DISTINCT "col")'
    )
    # Changed to List[str] with a default_factory to prevent NoneType errors
    entity_columns: List[str] = Field(
        default_factory=list, 
        description="List of group-by column names from the schema (e.g., ['store_id', 'sale_year_month'])"
    )
    filter_value: Optional[str] = Field(default=None, description="Specific value found in DB samples, or null")
    refined_instruction: str = Field(description="A clean technical instruction for a SQL coder")

    @field_validator("entity_columns", mode="before")
    @classmethod
    def clean_entities(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            # If LLM sent a single string, wrap it in a list
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
# Helpers
# ----------------------------
def _norm(s: str) -> str:
    """Normalizes strings for comparison by removing non-alphanumeric and lowercasing."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _extract_quoted_identifiers(expr: str) -> List[str]:
    """Extracts quoted identifiers from SQL expressions for validation."""
    return re.findall(r'"([^"]+)"', expr or "")


def _split_csvish(value: Any) -> List[str]:
    """Converts schema column strings back to lists for processing."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, (set, tuple)):
        return [str(x).strip() for x in list(value) if str(x).strip()]

    s = str(value).strip()
    if not s:
        return []

    # Split on comma, tolerate inconsistent spaces
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


# ----------------------------
# Query Interpreter (hard constraints)
# ----------------------------
class QueryInterpreter:
    """
    Interprets user queries into structured specs for SQL generation with schema constraints.
    """

    META_KEYS = {"TABLE", "PRIMARY_KEY_ID", "NUMERIC COLUMNS", "DATE COLUMNS", "TEXT COLUMNS"}

    # ---------- STEP 1: lightweight intent signals (generalizable) ----------
    _UNITS_HINTS = re.compile(r"\b(units|unit|quantity|qty)\b", re.IGNORECASE)
    _SOLD_HINTS = re.compile(r"\b(sold|sales)\b", re.IGNORECASE)
    _UNIQUE_HINTS = re.compile(r"\b(unique|distinct)\b", re.IGNORECASE)
    _AOV_HINTS = re.compile(r"\b(aov|average\s+order\s+value)\b", re.IGNORECASE)
    _TIME_HINTS = re.compile(r"\b(month|monthly|year|yearly|trend|over\s+time)\b", re.IGNORECASE)
    _NONEXISTENT_HINTS = re.compile(r"\b(doesn'?t\s+exist|non[-\s]?existent|not\s+in\s+data)\b", re.IGNORECASE)

    def __init__(self, con, table_name: str, type_schema: Dict[str, Any]):
        self.con = con
        self.table_name = table_name
        self.type_schema = type_schema

        # Parse the meta-schema into usable lists
        self.numeric_cols, self.date_cols, self.text_cols = self._parse_meta_schema(type_schema)

        # Entities = text + date (common group-by columns)
        self.entity_cols = list(dict.fromkeys(self.text_cols + self.date_cols))

        # "All columns" = numeric + entities (+ primary key if present)
        self.pk_col = str(type_schema.get("PRIMARY_KEY_ID") or "").strip() or None
        all_cols = []
        all_cols.extend(self.numeric_cols)
        all_cols.extend(self.entity_cols)
        if self.pk_col and self.pk_col not in all_cols:
            all_cols.append(self.pk_col)

        # Ground truth header set for constraints
        self.actual_headers: List[str] = list(dict.fromkeys(all_cols))
        self.norm_map: Dict[str, str] = {_norm(h): h for h in self.actual_headers}

    # ---------- meta schema parsing ----------
    def _parse_meta_schema(self, type_schema: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
        numeric_cols = _split_csvish(type_schema.get("NUMERIC COLUMNS"))
        date_cols = _split_csvish(type_schema.get("DATE COLUMNS"))
        text_cols = _split_csvish(type_schema.get("TEXT COLUMNS"))
        return numeric_cols, date_cols, text_cols

    # ---------- safety ----------
    def _escape_sql_like(self, s: str) -> str:
        return (s or "").replace("'", "''")

    # ---------- optional value context ----------
    def _get_data_context(self, user_query: str) -> Dict[str, List[str]]:
        """
        Provides sample values from text columns for query refinement.
        """
        keywords = re.findall(r"\b\w{4,}\b", user_query)
        ignore = {"what", "total", "highest", "lowest", "product", "limit", "define", "formula"}
        keywords = [w for w in keywords if w.lower() not in ignore]

        context: Dict[str, List[str]] = {}
        for col in self.text_cols[:3]:
            for word in keywords:
                try:
                    safe_word = self._escape_sql_like(word)
                    query = (
                        f'SELECT DISTINCT "{col}" '
                        f'FROM {self.table_name} '
                        f'WHERE CAST("{col}" AS TEXT) ILIKE \'%{safe_word}%\' '
                        f'LIMIT 3'
                    )
                    results = self.con.execute(query).fetchall()
                    if results:
                        context[col] = [str(r[0]) for r in results]
                except Exception:
                    continue
        return context

    # ---------- id column selection ----------
    def _pick_best_id_column(self) -> Optional[str]:
        """
        Selects the best ID column for unique counting.
        """
        if self.pk_col and self.pk_col in self.actual_headers:
            return self.pk_col

        headers = self.actual_headers
        hnorm = {_norm(h): h for h in headers}

        priority = [
            "transactionid", "transaction_id",
            "saleid", "sale_id",
            "invoiceid", "invoice_id",
            "orderid", "order_id",
            "receiptid", "receipt_id",
            "basketid", "basket_id",
        ]
        for p in priority:
            if p in hnorm:
                return hnorm[p]

        id_like = [h for h in headers if re.search(r"(^id$|_id$|\bid\b)", h.lower())]
        return id_like[0] if id_like else None

    # ---------- semantic matching ----------
    def _semantic_guess(self, guessed: str, candidates: List[str]) -> Optional[str]:
        """
        Maps user terms to schema columns using semantic anchors.
        """
        g = _norm(guessed)
        anchors = {
            "category": ["category", "cat"],
            "store": ["store", "shop", "location"],
            "price": ["price", "amount", "cost"],
            "quantity": ["quantity", "qty", "units"],
            "revenue": ["revenue", "sales", "earned"],
            "date": ["date", "time", "month", "year", "day", "trend"],
            "id": ["id", "sale", "invoice", "order", "transaction", "txn", "receipt", "basket"],
            "product": ["product", "item", "sku", "name"],
        }

        matched = None
        for k, words in anchors.items():
            if any(w in g for w in words):
                matched = k
                break
        if not matched:
            return None

        scored: List[Tuple[int, str]] = []
        for c in candidates:
            cn = _norm(c)
            score = 0
            if matched in cn:
                score += 3
            if matched == "category" and "id" in cn:
                score += 2
            if matched == "id" and ("id" in cn or cn.endswith("id")):
                score += 2
            if matched == "date" and ("month" in cn or "date" in cn or "year" in cn):
                score += 2
            scored.append((score, c))

        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[0][1] if scored and scored[0][0] > 0 else None

    # ---------- hard column coercion ----------
    def _coerce_to_schema_column(self, name: str, allowed: List[str]) -> Optional[str]:
        """
        Forces column names to match schema exactly.
        """
        if not name:
            return None

        # 1) exact
        if name in allowed:
            return name

        # 2) normalized exact
        n = _norm(name)
        if n in self.norm_map and self.norm_map[n] in allowed:
            return self.norm_map[n]

        # 3) semantic
        sem = self._semantic_guess(name, allowed)
        if sem:
            return sem

        # 4) fuzzy
        allowed_norm = {_norm(a): a for a in allowed}
        matches = get_close_matches(n, list(allowed_norm.keys()), n=1, cutoff=0.70)
        if matches:
            return allowed_norm[matches[0]]

        return None

    # ---------- STEP 2: deterministic rule layer (prevents the “units vs unique sales” bug) ---------- not generalized
    def _apply_domain_rules(self, spec: QuerySpec, user_query: str) -> QuerySpec:
        """
        Applies business rules to fix common query interpretation errors.
        """
        q = user_query or ""

        # ---- 2A) Units Sold should NEVER become COUNT(DISTINCT id) ----
        # Example failure you saw: "Monthly units sold per store" => COUNT(DISTINCT sale_id)
        wants_units = bool(self._UNITS_HINTS.search(q)) and bool(self._SOLD_HINTS.search(q))
        if wants_units:
            # pick a quantity-like numeric column from schema
            qty_col = self._coerce_to_schema_column("quantity", self.numeric_cols) or \
                      self._semantic_guess("quantity", self.numeric_cols)

            # If we have a quantity column, force SUM("quantity")
            if qty_col:
                spec.metric = f'SUM("{qty_col}")'
                # Helpful instruction for SQL generator downstream
                if self._TIME_HINTS.search(q):
                    spec.refined_instruction = "Aggregate total units sold by time bucket and the requested entity."
                else:
                    spec.refined_instruction = "Aggregate total units sold by the requested entity."

        # ---- 2B) Trend/time questions: force time-bucket entity if available ----
        # Example: CAT-3 trend 2021-2023 should group by sale_year_month (if present)
        if self._TIME_HINTS.search(q):
            # Preferred order of time buckets
            preferred_time_candidates = ["sale_year_month", "year_month", "month", "sale_date", "date"]
            
            time_col_found = None
            for cand in preferred_time_candidates:
                col = self._coerce_to_schema_column(cand, self.entity_cols)
                if col:
                    time_col_found = col
                    break
            
            if time_col_found:
                # Check if any time-related column is already in the list
                has_time_col = any(
                    re.search(r"(month|date|year)", c.lower()) 
                    for c in spec.entity_columns
                )
                
                if not has_time_col:
                    # APPEND the time column instead of overwriting
                    # This allows ['store_id', 'sale_year_month']
                    spec.entity_columns.append(time_col_found)

        # ---- 2C) AOV metric shaping (when user asks analytical AOV, not definition) ----
        # If query is like "highest AOV by store", you need a ratio.
        if self._AOV_HINTS.search(q) and any(w in q.lower() for w in ["highest", "lowest", "top", "rank", "by store", "per store", "store"]):
            # Find revenue-like and id-like columns
            rev_col = self._coerce_to_schema_column("revenue", self.numeric_cols) or self._semantic_guess("revenue", self.numeric_cols)
            id_col = self._pick_best_id_column()
            # If both exist, build AOV = SUM(revenue)/COUNT(DISTINCT id)
            if rev_col and id_col:
                spec.metric = f'(SUM("{rev_col}") / NULLIF(COUNT(DISTINCT "{id_col}"), 0)) AS aov'
                spec.refined_instruction = "Compute AOV as total revenue divided by distinct orders, grouped by the requested entity."
            # Also: if entity missing, default to store-ish entity
            if not spec.entity_columns:
                store_col = self._coerce_to_schema_column("store_id", self.entity_cols) or self._semantic_guess("store", self.entity_cols)
                if store_col:
                    spec.entity_columns = [store_col]

        # ---- 2D) "doesn't exist" filter should NOT become IS NULL ----
        # If the user explicitly says "product that doesn't exist", force an impossible literal.
        if self._NONEXISTENT_HINTS.search(q):
            # Try to filter by a product-ish column if any exist
            prod_col = self._coerce_to_schema_column("Product_Name", self.entity_cols) or \
                       self._semantic_guess("product", self.entity_cols)
            if prod_col:
                spec.filter_value = "__NON_EXISTENT__"  # safe sentinel that will not match
                spec.refined_instruction = f'Filter "{prod_col}" to a non-existent value and return the requested metric.'

        return spec

    def _coerce_metric(self, metric_expr: str) -> str:
        """
        Validates and maps metrics to schema columns.
        """
        m = (metric_expr or "").strip()
        if not m:
            raise ValueError("Metric is empty.")

        # aggregate/function/expression
        if "(" in m:
            quoted = _extract_quoted_identifiers(m)

            for q in quoted:
                if q not in self.actual_headers:
                    # Special fix: "transaction_id" -> real id col
                    if _norm(q) in ("transactionid", "transaction_id"):
                        id_col = self._pick_best_id_column()
                        if not id_col:
                            raise ValueError('No suitable ID column found in schema to replace "transaction_id".')
                        m = m.replace(f'"{q}"', f'"{id_col}"')
                        continue

                    fixed_inner = self._coerce_to_schema_column(q, self.actual_headers)
                    if not fixed_inner:
                        raise ValueError(f"Invalid metric column referenced inside function: {q}")
                    m = m.replace(f'"{q}"', f'"{fixed_inner}"')

            return m

        # simple numeric metric column
        fixed = self._coerce_to_schema_column(m, self.numeric_cols)
        if not fixed:
            raise ValueError(f"Invalid metric: '{m}'. Must be one of numeric columns: {self.numeric_cols[:12]}...")
        return fixed

    def _hard_enforce(self, spec: QuerySpec) -> QuerySpec:
        """Enforces schema compliance on query specs."""
        # 1. Validate the metric remains schema-grounded
        spec.metric = self._coerce_metric(spec.metric)

        # 2. ✅ FIX: Update 'entity_column' to 'entity_columns'
        if not spec.entity_columns:
            # Provide a default grouping if the LLM left it empty
            default_id = self.pk_col or self._pick_best_id_column() or "sale_id"
            spec.entity_columns = [default_id]

        # 3. ✅ FIX: Loop through the list to validate each column
        validated_entities = []
        for col in spec.entity_columns:
            fixed = self._coerce_to_schema_column(col, self.entity_cols)
            if fixed:
                validated_entities.append(fixed)

        # 4. Error handling if no valid columns were found
        if not validated_entities:
            raise ValueError(f"No valid entity columns found in: {spec.entity_columns}")
        
        # 5. Deduplicate and update the spec
        spec.entity_columns = list(dict.fromkeys(validated_entities)) 
        
        return spec

    # ---------- main ----------
    def refine_intent(self, user_query: str) -> QuerySpec:
        """Main method to refine user query into validated QuerySpec."""
        llm = ChatOllama(model="qwen2.5:7b-instruct", format="json", temperature=0)
        parser = PydanticOutputParser(pydantic_object=QuerySpec)

        data_context = self._get_data_context(user_query)

        # ---------- STEP 3: tighten the prompt so LLM stops inventing the wrong metric ----------
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a precise Data Architect. Map the user query to the schema EXACTLY.

AVAILABLE COLUMNS (METRICS): {metrics}
AVAILABLE COLUMNS (ENTITIES): {entities}
PRIMARY KEY (if any): {primary_key}
SAMPLE VALUES (optional hints): {data_context}

HARD RULES:
1) You MUST ONLY use column names from the lists above.
2) If user requests "unique" transactions/orders/sales -> use COUNT(DISTINCT "<id column from schema>").
3) If user requests "units sold" or "quantity sold" -> use SUM("<quantity-like numeric column>"), NOT COUNT(DISTINCT ...).
4) If user asks for "trend/monthly/yearly" -> entity_column should be a time bucket (prefer something like sale_year_month if present).
5) If user asks for AOV in an analytical sense (highest/lowest/by store), you may express it as:
   (SUM("revenue") / NULLIF(COUNT(DISTINCT "<id>"),0)) AS aov  (ONLY if those columns exist in schema lists).
6) Never invent names like "transaction_id" unless it exists in schema lists.
7) Return JSON ONLY that matches the provided output schema.
{format_instructions}"""),
            ("human", "{query}")
        ])

        chain = prompt | llm | parser
        spec = chain.invoke({
            "metrics": self.numeric_cols,
            "entities": self.entity_cols,
            "primary_key": self.pk_col or "None",
            "data_context": data_context,
            "query": user_query,
            "format_instructions": parser.get_format_instructions()
        })

        # ✅ Apply deterministic fixes first (prevents your known errors)
        spec = self._apply_domain_rules(spec, user_query)

        # ✅ Then hard-enforce schema grounding
        return self._hard_enforce(spec)