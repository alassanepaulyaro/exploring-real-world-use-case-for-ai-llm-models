# Summarize with Mistral + Streamlit + Ollama
import streamlit as st
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"  # Change to your desired Ollama model

st.set_page_config(page_title="Ollama Summarizer", layout="centered")
st.title("📝 Text Summarizer with Ollama")

st.markdown(
    """
    Enter your text below and get a concise summary using the **Mistral** model via your local Ollama server.
    """
)

with st.form("summarize_form", clear_on_submit=False):
    user_text = st.text_area("Enter text to summarize:", height=200)
    submitted = st.form_submit_button("Summarize")

if submitted and user_text.strip():
    with st.spinner("Summarizing..."):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": f"Summarize this: {user_text}",
                    "stream": False
                },
                headers={"Content-Type": "application/json"}
            )
            response_data = response.text.strip()
            try:
                json_response = json.loads(response_data)
                summary = json_response.get("response", "No valid summary received.")
            except json.JSONDecodeError:
                summary = f"❌ Invalid JSON response from Ollama: {response_data}"
        except requests.exceptions.RequestException as e:
            summary = f"❌ Request to Ollama failed: {str(e)}"
    st.markdown("#### Summary:")
    st.success(summary)
elif submitted:
    st.warning("Please enter text to summarize.")

st.markdown("---")
st.caption("Powered by Ollama + Mistral | [Learn more about Ollama](https://ollama.com/)")

