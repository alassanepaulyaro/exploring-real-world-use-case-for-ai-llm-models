# Grammar & Spell Checker (Ollama + Mistral)
import streamlit as st
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral:latest"  # Best choice for proofreading with Ollama

st.set_page_config(page_title="Ollama Proofreader (Mistral)", layout="centered")
st.title("📝 Grammar & Spell Checker (Ollama + Mistral)")

st.markdown("""
Enter the text you want to proofread (grammar, spelling, and sentence structure corrections will be applied).
""")

with st.form("proofread_form", clear_on_submit=False):
    user_text = st.text_area("Text to proofread", height=220)
    submitted = st.form_submit_button("Proofread")

if submitted and user_text.strip():
    prompt = f"Correct the grammar, spelling, and sentence structure of the following text:\n{user_text}"
    with st.spinner("Proofreading with Mistral..."):
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
                    proofread_text = json_response["response"]
                elif "error" in json_response:
                    st.error(f"Ollama error: {json_response['error']}")
                    proofread_text = ""
                else:
                    st.error(f"⚠️ Unexpected Ollama API response: {json_response}")
                    proofread_text = ""
            except json.JSONDecodeError:
                st.error(f"Invalid JSON from Ollama: {response_data}")
                proofread_text = ""
        except requests.exceptions.RequestException as e:
            st.error(f"Request to Ollama failed: {str(e)}")
            proofread_text = ""
    if proofread_text:
        st.markdown("#### Proofread Text:")
        st.text_area("Corrected text", value=proofread_text, height=220, key="corrected_area")
        st.info("You can copy and reuse the corrected text above.")
elif submitted:
    st.warning("Please enter text to proofread.")

st.markdown("---")
st.caption("Powered by Ollama + Mistral")


