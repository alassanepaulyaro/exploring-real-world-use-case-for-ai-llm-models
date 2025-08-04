# AI-Powered Sentiment Analysis
import streamlit as st
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral-nemo:latest"

def analyze_sentiment(text):
    """
    Uses Mistral-Nemo (Ollama) to classify sentiment and highlight contributing words.
    """
    prompt = (
        "Classify the sentiment of the following text as Positive, Negative, or Neutral. "
        "Then, list the words or expressions from the text that contribute to the detected sentiment. "
        "If the text is in French, answer in French. Otherwise, answer in English.\n\n"
        f"Text:\n{text}\n\n"
        "Sentiment and Highlights:"
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "No sentiment detected.")
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Request error: {e}"

st.set_page_config(page_title="AI Sentiment Analysis (Mistral-Nemo)", layout="centered")
st.title("🔎 AI-Powered Sentiment Analysis")

st.markdown("""
Enter a sentence or paragraph below.  
**Mistral-Nemo** (via Ollama) will classify the sentiment (**Positive**, **Negative**, or **Neutral**)  
and highlight the words that contribute to that sentiment.
""")

with st.form(key="sentiment_form"):
    text = st.text_area(
        "Text to analyze",
        height=100,
        max_chars=1000,
        placeholder="Enter a sentence for sentiment analysis…"
    )
    submit = st.form_submit_button("Analyze Sentiment")

if submit and text.strip():
    with st.spinner("Analyzing sentiment..."):
        result = analyze_sentiment(text)
        st.subheader("Sentiment Result")
        st.markdown(result)

st.markdown("""
---
<small>Powered by Mistral-Nemo, Ollama, and Streamlit</small>
""", unsafe_allow_html=True)
