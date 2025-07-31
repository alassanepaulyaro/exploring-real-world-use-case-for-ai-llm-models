# Content Writing with LLaMA 3 + Ollama
import streamlit as st
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"  # Use your pulled Ollama model here

st.set_page_config(page_title="Ollama Blog Post Generator", layout="centered")
st.title("📝 Blog Post Generator with Ollama")

st.markdown("""
Enter a **topic** and choose a **writing style** to generate a blog post using your local Ollama LLM.
""")

with st.form("generate_form", clear_on_submit=False):
    topic = st.text_input("Topic", value="")
    style = st.selectbox("Writing Style", ["informative", "casual", "persuasive", "humorous", "technical"])
    submitted = st.form_submit_button("Generate")

if submitted and topic and style:
    prompt = f"Write a detailed blog post about '{topic}' in a {style} tone."
    with st.spinner("Generating content..."):
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
                    generated_content = json_response["response"]
                elif "error" in json_response:
                    st.error(f"Ollama error: {json_response['error']}")
                    generated_content = ""
                else:
                    st.error(f"⚠️ Unexpected Ollama API response: {json_response}")
                    generated_content = ""
            except json.JSONDecodeError:
                st.error(f"Invalid JSON from Ollama: {response_data}")
                generated_content = ""
        except requests.exceptions.RequestException as e:
            st.error(f"Request to Ollama failed: {str(e)}")
            generated_content = ""
    if generated_content:
        st.markdown("#### Generated Blog Post:")
        st.success(generated_content)
elif submitted:
    st.warning("Please provide a topic and select a writing style.")

st.markdown("---")
st.caption("Powered by Ollama | [Learn more about Ollama](https://ollama.com/)")
