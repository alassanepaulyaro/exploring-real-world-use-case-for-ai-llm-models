# AI Legal Document Analyzer + Ollama + Phi

import streamlit as st
import requests
import json
from io import StringIO

try:
    import docx
except ImportError:
    st.error("Please install python-docx: pip install python-docx")
    st.stop()

try:
    import PyPDF2
except ImportError:
    st.error("Please install PyPDF2: pip install PyPDF2")
    st.stop()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi4:latest"  # Change to your preferred model

st.set_page_config(page_title="Legal Doc Analyzer (Ollama + Phi)", layout="centered")
st.title("📄 Legal Document Analyzer (Ollama + Phi)")

st.markdown("""
**Upload a legal document** (`.txt`, `.pdf`, or `.docx`), or paste text below.  
Ollama will extract and summarize key clauses, risks, and obligations.
""")

uploaded_file = st.file_uploader("Choose a document", type=["txt", "pdf", "docx"])

file_text = ""
if uploaded_file is not None:
    filetype = uploaded_file.name.split(".")[-1].lower()
    try:
        if filetype == "txt":
            file_text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        elif filetype == "pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            pages = []
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            file_text = "\n".join(pages)
        elif filetype == "docx":
            doc = docx.Document(uploaded_file)
            file_text = "\n".join([p.text for p in doc.paragraphs])
        else:
            st.error("Unsupported file type.")
    except Exception as e:
        st.error(f"Failed to extract text: {str(e)}")

user_text = st.text_area("Or paste legal document text", value=file_text, height=300, key="user_text_area")

with st.form("analyze_legal_text", clear_on_submit=False):
    submitted = st.form_submit_button("Analyze")

if submitted and user_text.strip():
    prompt = (
        f"Extract key insights from the following legal document:\n"
        f"{user_text}\n"
        "Summarize important clauses, risks, and obligations."
    )
    with st.spinner("Analyzing legal document..."):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False
                },
                headers={"Content-Type": "application/json"}
            )
            response_data = response.text.strip()
            try:
                json_response = json.loads(response_data)
                if "response" in json_response and json_response["response"]:
                    legal_insights = json_response["response"]
                elif "error" in json_response:
                    st.error(f"Ollama error: {json_response['error']}")
                    legal_insights = ""
                else:
                    st.error(f"⚠️ Unexpected Ollama API response: {json_response}")
                    legal_insights = ""
            except json.JSONDecodeError:
                st.error(f"Invalid JSON from Ollama: {response_data}")
                legal_insights = ""
        except requests.exceptions.RequestException as e:
            st.error(f"Request to Ollama failed: {str(e)}")
            legal_insights = ""
    if legal_insights:
        st.markdown("#### Key Insights / Summary:")
        st.text_area("Extracted Insights", value=legal_insights, height=220, key="insights_area")
        st.info("Review, edit, or copy the extracted insights above.")
elif submitted:
    st.warning("Please upload a document or paste legal text to analyze.")

st.markdown("---")
st.caption("Powered by Ollama + Phi")

