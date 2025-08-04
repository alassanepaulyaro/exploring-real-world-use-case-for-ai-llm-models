# Medical AI Symptom Analyzer
import streamlit as st
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "medllama2:7b"

st.set_page_config(page_title="AI Symptom Analyzer", layout="centered")
st.title("🩺 Medical AI Symptom Analyzer")
st.write("""
Enter your symptoms below.  
The AI will provide possible explanations and general advice.  
**No diagnosis. This does not replace a doctor's consultation.**
""")

with st.form(key="symptom_form"):
    symptoms = st.text_area("Describe your symptoms", height=120, max_chars=1000)
    submit = st.form_submit_button("Analyze Symptoms")

if submit and symptoms.strip():
    with st.spinner("Analyzing symptoms..."):
        prompt = f"""You are a medical AI assistant trained to analyze symptoms.
        Based on the provided symptoms, give possible explanations and general advice.
        Do not provide a diagnosis or replace a doctor's consultation.

        User Symptoms: {symptoms}

        Medical AI:"""

        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(
                OLLAMA_URL,
                json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
                headers=headers,
                timeout=90
            )
            response_data = response.text.strip()
            try:
                json_response = json.loads(response_data)
                ai_response = json_response.get("response", "Sorry, I couldn't generate a response.")
            except Exception:
                st.error(f"Invalid response from Ollama: {response_data}")
                ai_response = None

        except Exception as e:
            st.error(f"Request to Ollama failed: {e}")
            ai_response = None

        if ai_response:
            st.subheader("AI Advice")
            st.markdown(ai_response)

st.markdown("""
---
<small>Powered by MedLLaMA 2 via Ollama - For informational purposes only.</small>
""", unsafe_allow_html=True)
