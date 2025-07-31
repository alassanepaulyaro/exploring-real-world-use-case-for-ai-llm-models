# Code Generator and Debugger with Code Llama
import streamlit as st
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"  # Use your pulled Ollama CodeLlama model

st.set_page_config(page_title="Ollama Code Generator & Debugger", layout="wide")

# CSS for responsive and resizable view
st.markdown("""
<style>
.main-container {
    display: flex;
    height: 80vh;
    gap: 10px;
}
.chat-panel {
    flex: 1;
    min-width: 300px;
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 15px;
    overflow-y: auto;
}
.canvas-panel {
    flex: 1;
    min-width: 300px;
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 15px;
    display: flex;
    flex-direction: column;
}
.canvas-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.canvas-content {
    flex: 1;
    border: 1px solid #ccc;
    border-radius: 3px;
    padding: 10px;
    background-color: #f8f9fa;
    font-family: 'Courier New', monospace;
    font-size: 14px;
    overflow-y: auto;
    white-space: pre-wrap;
}
.resize-handle {
    width: 5px;
    background-color: #ddd;
    cursor: col-resize;
    border-radius: 2px;
}
.resize-handle:hover {
    background-color: #007bff;
}
@media (max-width: 768px) {
    .main-container {
        flex-direction: column;
        height: auto;
    }
    .resize-handle {
        display: none;
    }
}
</style>
""", unsafe_allow_html=True)

st.title("💻 Code Generator & Debugger (Ollama + CodeLlama)")

# Initialize session state for canvas
if 'canvas_content' not in st.session_state:
    st.session_state.canvas_content = ""
if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = ""
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = ""

# Layout principal
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 💬 Chat Panel")
    
    with st.form("code_form", clear_on_submit=True):
        mode = st.radio("Mode", ["generate", "debug"], horizontal=True)
        prompt = st.text_area(
            "Prompt (for generate: describe what you want, for debug: paste code to fix)",
            height=200
        )
        submitted = st.form_submit_button("Run")
    
    # Display current processing status
    if st.session_state.processing_status:
        st.success(st.session_state.processing_status)
        if st.button("Clear Status", key="clear_status"):
            st.session_state.processing_status = ""
            st.rerun()
    
    # Display current prompt being processed
    if st.session_state.current_prompt:
        st.markdown("#### Current Request:")
        st.info(f"Processing: {st.session_state.current_prompt[:100]}...")

with col2:
    st.markdown("### 📝 Canvas Panel")
    
    # Header du canvas avec boutons
    col_header1, col_header2, col_header3 = st.columns([2, 1, 1])
    
    with col_header2:
        if st.button("📋 Copy", key="copy_btn"):
            st.toast("Code copied to clipboard!")
    
    with col_header3:
        if st.button("🗑️ Clear", key="clear_btn"):
            st.session_state.canvas_content = ""
            st.rerun()
    
    # Zone de texte modifiable pour le canvas
    canvas_content = st.text_area(
        "Canvas Content (editable)",
        value=st.session_state.canvas_content,
        height=400,
        key="canvas_editor",
        help="You can modify the code directly here"
    )
    
    # Mettre à jour le contenu du canvas
    if canvas_content != st.session_state.canvas_content:
        st.session_state.canvas_content = canvas_content

# Process form submission
if submitted and prompt.strip():
    # Clear previous status and set current prompt
    st.session_state.processing_status = ""
    st.session_state.current_prompt = prompt
    
    if mode == "generate":
        full_prompt = f"Write a clean, well-documented {prompt} code snippet."
    elif mode == "debug":
        full_prompt = f"Debug and fix the following code:\n{prompt}"
    else:
        st.error("Invalid mode selected.")
        full_prompt = None
    
    if full_prompt:
        with st.spinner("Processing..."):
            try:
                response = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": MODEL_NAME,
                        "prompt": full_prompt,
                        "stream": False
                    },
                    headers={"Content-Type": "application/json"}
                )
                response_data = response.text.strip()
                try:
                    json_response = json.loads(response_data)
                    if "response" in json_response and json_response["response"]:
                        code_result = json_response["response"]
                        # Update canvas with result
                        st.session_state.canvas_content = code_result
                        # Clear chat panel by clearing current prompt and setting success status
                        st.session_state.current_prompt = ""
                        st.session_state.processing_status = f"✅ Successfully {mode}d code! Check the canvas panel."
                        st.rerun()
                    elif "error" in json_response:
                        st.error(f"Ollama error: {json_response['error']}")
                        st.session_state.processing_status = "❌ Error occurred during processing."
                    else:
                        st.error(f"⚠️ Unexpected Ollama API response: {json_response}")
                        st.session_state.processing_status = "⚠️ Unexpected response from Ollama."
                except json.JSONDecodeError:
                    st.error(f"Invalid JSON from Ollama: {response_data}")
                    st.session_state.processing_status = "❌ Invalid response format."
            except requests.exceptions.RequestException as e:
                st.error(f"Request to Ollama failed: {str(e)}")
                st.session_state.processing_status = "❌ Connection to Ollama failed."

elif submitted:
    st.warning("Please enter a prompt or code.")
    st.session_state.processing_status = "⚠️ Please enter a prompt or code."

# JavaScript for copy functionality
st.markdown("""
<script>
function copyToClipboard() {
    const textArea = document.querySelector('textarea[aria-label="Canvas Content (editable)"]');
    if (textArea) {
        textArea.select();
        document.execCommand('copy');
    }
}
</script>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("Powered by Ollama + CodeLlama | [Learn more about Ollama](https://ollama.com/)")