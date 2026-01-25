import duckdb
from pathlib import Path
import re

def load_csv_to_duckdb(csv_path: str):
    """
    Loads CSV into DuckDB and extracts schema for SQL operations.
    """
    con = duckdb.connect()
    table_name = Path(csv_path).stem.lower()
    table_name = re.sub(r'[^a-z0-9_]', '_', table_name)
    ingestion_warnings = [] # Collect data quality warnings
    # Load CSV into DuckDB
    con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM read_csv_auto('{csv_path}')")

    # Extract schema information
    schema_info = con.execute(f"DESCRIBE {table_name}").fetchall()
    # Auto-detect and fix "Dirty Number" columns
    for col in schema_info:
        name, dtype = col[0], str(col[1]).upper()
        
        # Only check text-like columns
        if "VARCHAR" in dtype or "STRING" in dtype:
            # Check if 80% of non-null values match a "Dirty Number" pattern (Currency, Commas)
            # Regex: Optional currency ($/£/€), digits, optional commas, optional decimals
            sample_query = f"""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN regexp_matches("{name}", '^[\$£€¥]?\s*-?([0-9]{{1,3}}(,[0-9]{{3}})*|[0-9]+)(\.[0-9]+)?$') THEN 1 END) as matches
                FROM {table_name}
                WHERE "{name}" IS NOT NULL AND "{name}" != ''
            """
            try:
                total, matches = con.execute(sample_query).fetchone()
                if total > 0 and (matches / total) > 0.8:
                    print(f"--- AUTO-FIX: Converting dirty column '{name}' to DOUBLE ---")
                    # Calculate dirty percentage for warning
                    clean_ratio = matches / total
                    dirty_pct = (1.0 - clean_ratio) * 100
                    # [NEW] Add explicit warning if data is messy
                    if dirty_pct > 0:
                        ingestion_warnings.append(
                            f"Data Quality Alert: Column '{name}' was auto-converted to numeric, but {dirty_pct:.1f}% of values were unreadable (text/symbols) and excluded."
                        )
                    # Clean and Cast in place
                    # 1. Remove currency symbols and commas
                    # 2. Cast to DOUBLE
                    con.execute(f"""
                        ALTER TABLE {table_name} ALTER COLUMN "{name}" TYPE DOUBLE 
                        USING TRY_CAST(regexp_replace("{name}", '[^0-9\.-]', '', 'g') AS DOUBLE)
                    """)
            except Exception as e:
                print(f"Type Check Warning for {name}: {e}")

   # 3. Final Schema Extraction (Now accurate)
    final_schema = con.execute(f"DESCRIBE {table_name}").fetchall()
    numeric_cols, date_cols, text_cols = [], [], []
    
    # Identifies primary key for transaction uniqueness.
    pk_candidate = None
    cols_found = [col[0].lower() for col in final_schema]
    
    for preferred in ["sale_id", "transaction_id", "order_id", "id"]:
        if preferred in cols_found:
            pk_candidate = preferred
            break
    
    if not pk_candidate: # Fallback to any column ending in _id
        for col in cols_found:
            if col.endswith("_id"):
                pk_candidate = col
                break

    for col in final_schema:
        name, dtype = col[0], str(col[1]).upper()
        if any(t in dtype for t in ["INT", "DOUBLE", "FLOAT", "DECIMAL"]):
            numeric_cols.append(name)
        elif any(t in dtype for t in ["DATE", "TIMESTAMP"]):
            date_cols.append(name)
        else:
            text_cols.append(name)

    type_aware_schema = {
    "TABLE": table_name,
    "PRIMARY_KEY_ID": pk_candidate,
    "NUMERIC COLUMNS": ", ".join(numeric_cols),
    "DATE COLUMNS": ", ".join(date_cols),
    "TEXT COLUMNS": ", ".join(text_cols),
    }

    return con, table_name, type_aware_schema, numeric_cols , ingestion_warnings