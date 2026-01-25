# ui.py
# Run: streamlit run Code/ui.py

import time
import requests
import streamlit as st
import json
import plotly.io as pio 
import uuid # [FIX] Required for generating unique keys

API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(page_title="BI Agent", page_icon="📊", layout="wide")
st.title("BI Agent")

# -----------------------------
# Helpers
# -----------------------------
def typewriter(text: str, speed: float = 0.012):
    """Provides typewriter animation for assistant responses."""
    placeholder = st.empty()
    rendered = ""
    for ch in text:
        rendered += ch
        placeholder.markdown(rendered)
        time.sleep(speed)
    return rendered

def call_api(question: str, csv_file, pdf_files):
    """Sends user query and uploaded files to the backend API."""
    multipart_files = [
        ("csv_file", (csv_file.name, csv_file.getvalue(), "text/csv"))
    ]
    for p in (pdf_files or []):
        multipart_files.append(("pdf_files", (p.name, p.getvalue(), "application/pdf")))

    data = {"question": question}
    return requests.post(API_URL, data=data, files=multipart_files, timeout=600)

# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "intro_done" not in st.session_state:
    st.session_state.intro_done = False

# -----------------------------
# Sidebar uploads
# -----------------------------
with st.sidebar:
    st.header("Data")
    csv_file = st.file_uploader("Drop your CSV", type=["csv"])
    pdf_files = st.file_uploader("Drop PDFs (optional)", type=["pdf"], accept_multiple_files=True)
    st.caption("Upload once, then just chat.")

# -----------------------------
# Intro message (typed once)
# -----------------------------
INTRO = (
    "Hey 👋 I’m your BI Agent.\n\n"
    "1) Upload a **CSV** in the sidebar (PDFs are optional).\n"
    "2) Ask me questions like:\n"
    "   - *Top 5 of certain metrics*\n"
    "   - *Sales trend analysis*\n"
    "   - *Descriptive calculations*\n\n"
    "I’ll use docs for definitions + rules, and SQL when calculations are needed."
)

if not st.session_state.intro_done and len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        typewriter(INTRO, speed=0.010)

    st.session_state.messages.append({"role": "assistant", "content": INTRO})
    st.session_state.intro_done = True
    st.rerun()

# -----------------------------
# Render chat history
# -----------------------------
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        # [FIX] Check if this message has a chart attached and render it with a UNIQUE KEY
        if m.get("viz_data"):
            try:
                # Convert JSON string back to Plotly Figure
                fig = pio.from_json(m["viz_data"])
                # We use the index 'i' to guarantee a unique key for history items
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}")
            except Exception as e:
                st.error(f"Error rendering chart: {e}")

# -----------------------------
# Chat input
# -----------------------------
question = st.chat_input("Ask a business question...")

if question:
    if not csv_file:
        st.error("Upload a CSV in the sidebar first.")
        st.stop()

    # show user message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # call backend + typewriter response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            r = call_api(question, csv_file, pdf_files)

        if r.status_code != 200:
            err = f"API error {r.status_code}: {r.text}"
            typewriter(err, speed=0.008)
            st.session_state.messages.append({"role": "assistant", "content": err})
            st.stop()

        out = r.json()
        answer = (out.get("final_answer") or "").strip() or "(No answer returned.)"
        
        # Extract viz data
        viz_json = out.get("viz_data")

        # 1. Render Text
        typewriter(answer, speed=0.010)
        
        # 2. Render Chart (if exists)
        if viz_json:
            try:
                fig = pio.from_json(viz_json)
                # [FIX] Generate a random UUID for the new chart to prevent ID collisions
                st.plotly_chart(fig, use_container_width=True, key=f"new_chart_{uuid.uuid4()}")
            except Exception as e:
                st.error(f"Could not render chart: {e}")

    # Save both text and viz_data to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer, 
        "viz_data": viz_json
    })
    
    st.rerun()