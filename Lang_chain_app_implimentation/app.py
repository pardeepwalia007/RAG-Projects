# original code router no llm router  this is updated and will be tested with generalized intent llm what uses business context. 
from ingestion import  ingest_files
from sql_engine import load_csv_to_duckdb
from pdf_to_markdown import pdfs_to_markdown
from vectorize import build_retriever
from sql_orchestrator import should_run_sql
from summarization_agent import summarize_with_llama
from tests_logger import log_test
# from query_refiner import QueryInterpreter
from llm_sql_agent import sql_pipeline_structured
import logging
from intent_llm import QueryInterpreter
NOISY_LOGGERS = [
    "httpx",
    "urllib3",
    "ollama",
    "chromadb",
    "langchain",
    "langchain_core",
]

for logger in NOISY_LOGGERS:
    logging.getLogger(logger).setLevel(logging.CRITICAL)

logging.disable(logging.CRITICAL)


"""
Orchestration Layer of all the functionality

"""

def main():
    # loaded paths of csv and pdfs
    paths = [r"/Users/pardeepwalia/Desktop/Data/Agentic_RAG/Data/docs/Business_Metrics_Detailed.pdf",r"/Users/pardeepwalia/Desktop/Data/Agentic_RAG/Data/csv/Sales_enriched.csv",r"/Users/pardeepwalia/Desktop/Data/Agentic_RAG/Data/docs/Business_Rules_Detailed.pdf"]
    
    # deconstructs csv and pdf[list] from ingested file : validates and run through a piple line to give seperated path for valid csv and pdfs 
    csv_path,pdf_paths = ingest_files(paths)

    #  sql query using duckdb
    con,table_name,type_schema,num_cols = load_csv_to_duckdb(csv_path)

    # Pdf Section 
    paths_md,errors_md,is_md= pdfs_to_markdown(pdf_paths,r"/Users/pardeepwalia/Desktop/Data/Agentic_RAG/Data/docs/")
    
    # Retriever 
    retriever=build_retriever(paths_md)

    # Testing the query
    test_cases = [
        #  "Sales trend between Jan and Dec of 2023",
        # "Top 5 categories by number of unique transactions",
        # "Monthly units sold per store",
        # "Total transactions in 2021",
        # "How is AOV calculated?",
        # "Define total orders",
        # "What are the prohibited analyses?",
        # "Can we compute churn?",
        # # strech tests
        "What is the revenue trend for CAT-3 between 2021 and 2023",
        # "Total transactions for a product that doesn't exist"
    ]
    # for q in test_cases:
    print("🧠 Agentic RAG Chat — type 'exit' to quit.\n")
    # for q in test_cases:
    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("👋 Bye!")
            break
        print(f"User: {q} ")
        print("\n📖 ACTION: Retrieving Business Context...")
        chunks = retriever.invoke(q)
        doc_evidence = "\n".join(c.page_content for c in chunks) if chunks else ""

        run_sql = should_run_sql(q)
        output = None

        #  this will change 
        if run_sql:
            interpreter = QueryInterpreter(con, table_name, type_schema)
            refined_spec = interpreter.refine_intent(q,business_context=doc_evidence)
            print(f"intent: {refined_spec}")
            print("🚀 ACTION: Routing to SQL Engine (Data Calculation needed)")
            output = sql_pipeline_structured(q,refined_spec,con=con,table_name=table_name,type_schema=type_schema)
            print(f"sql output: {output.get('sql')}")
        # this will chage 
        summary_payload = {
            "question": q,
            "source_type": "Hybrid (Docs + SQL)" if run_sql else "Docs Only",
            "business_rules": doc_evidence if doc_evidence else None,
            "sql_output": output if run_sql else None,
        }

        # this reamins same
        final_answer = summarize_with_llama(
            question=q,
            evidence=summary_payload,
            source_type=summary_payload["source_type"],
        )
        log_test(q,final_answer)
        print("\n🤖 Agent Response:")
        print(final_answer)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()