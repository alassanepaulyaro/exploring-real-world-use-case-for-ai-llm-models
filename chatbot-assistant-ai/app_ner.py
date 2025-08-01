# AI-Powered Named Entity Recognition (NER)
import streamlit as st
import requests

# --- Available models ---
MODELS = [
    "mistral-nemo:latest",
    "hf.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q3_K_S",
    "qwen3:14b",
    "medllama2:7b",
    "granite3.3:latest",
    "llama3.1:8b",
    "deepseek-r1:latest",
]

OLLAMA_URL = "http://localhost:11434/api/generate"

def extract_named_entities(text, model):
    """
    Uses the selected model to extract named entities (persons, organizations, locations, dates).
    The prompt supports both English and French inputs.
    """
    prompt = (
        "Extract all named entities from the following text. "
        "List each entity under its category: Persons, Organizations, Locations, Dates. "
        "If the text is in French, answer in French. Otherwise, answer in English.\n\n"
        f"Text:\n{text}\n\n"
        "Entities:"
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=90)
        if response.status_code == 200:
            return response.json().get("response", "No entities detected.")
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Request error: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="AI NER Multi-Model (Ollama)", layout="centered")
st.title("🧠 AI-Powered Named Entity Recognition (NER)")

st.markdown("""
- Paste a paragraph (English or French).
- Select a model (recommended: *Mistral-Small-24B-Instruct*, *Mistral-Nemo*, *Qwen3-14B*, or *MedLLaMA2* for medical).
- Receive structured named entities: **persons, organizations, locations, dates**.
""")

with st.form("ner_form"):
    text = st.text_area("Text to analyze", height=140, max_chars=3000, placeholder="Paste your text here…")
    model = st.selectbox("Select model", MODELS, index=0)
    submit = st.form_submit_button("Extract Entities")

if submit and text.strip():
    with st.spinner(f"Extracting entities with {model}…"):
        result = extract_named_entities(text, model)
        st.subheader("Extracted Entities")
        st.code(result)

st.markdown("---")
st.markdown("""
<small>Powered by Ollama & Streamlit • NER Multi-Model</small>
""", unsafe_allow_html=True)
