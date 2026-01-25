import json
from pathlib import Path

LOG_FILE = Path("/Users/pardeepwalia/Desktop/Data/Agentic_RAG/tests/test_logs_langgraph1.json")

def log_test(question: str, response: str):
    entry = {
        "question": question,
        "response": response
    }

    # If file exists, load → append → save
    if LOG_FILE.exists():
        with LOG_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with LOG_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)