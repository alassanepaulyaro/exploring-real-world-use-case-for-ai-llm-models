# PDF Summarizer with Qwen3
import streamlit as st
import fitz  # PyMuPDF
import requests
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import os
import json
import time

# Ollama API URL
OLLAMA_URL = "http://localhost:11434/api/generate"

# Set page config
st.set_page_config(
    page_title="PDF Summarizer with Qwen3",
    page_icon="📄",
    layout="centered"
)

st.title("📄 PDF Summarizer with Qwen3 (via Ollama)")
st.markdown("Upload a PDF to extract and summarize using **Qwen3**.")

# Temporary file path
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)
TEMP_PATH = os.path.join(TEMP_DIR, "uploaded.pdf")

# Function: Extract text using PyMuPDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            extracted = page.get_text("text")
            if extracted:
                text += extracted + "\n"
        doc.close()
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {e}"

# Function: OCR for scanned PDFs
def extract_text_with_ocr(pdf_path):
    try:
        images = convert_from_path(pdf_path)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
        return text.strip()
    except Exception as e:
        return f"OCR failed: {e}"

# Function: Summarize text using Ollama with qwen3 (streaming fix)
def summarize_text_stream(prompt, model="qwen3:latest"):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True) as response:
            if response.status_code != 200:
                yield f"❌ Error {response.status_code}: {response.text}"
                return

            for line in response.iter_lines():
                if line:
                    try:
                        # Parse JSON response from Ollama
                        body = line.decode('utf-8')
                        data = json.loads(body)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done"):  # End of stream
                            break
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        yield f"[Error parsing stream: {e}]"
                        continue
    except requests.ConnectionError:
        yield "❌ Connection failed. Is Ollama running? Run `ollama serve` in terminal."
    except Exception as e:
        yield f"❌ Unexpected error: {e}"

# Sidebar options
st.sidebar.header("🔧 Settings")
model_choice = st.sidebar.selectbox(
    "Choose Qwen3 Model",
    ("qwen3:latest", "qwen3:14b"),
    index=0
)

use_ocr = st.sidebar.checkbox("Force OCR (for scanned PDFs)", False)
max_chars = st.sidebar.slider("Max characters sent to AI", 1000, 32000, 8000)

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    # Save uploaded PDF
    with open(TEMP_PATH, "wb") as f:
        f.write(uploaded_file.getvalue())

    with st.spinner("🔍 Extracting text from PDF..."):
        if use_ocr:
            raw_text = extract_text_with_ocr(TEMP_PATH)
        else:
            raw_text = extract_text_from_pdf(TEMP_PATH)
            # Fallback to OCR if little text found
            if not raw_text or len(raw_text.strip()) < 100:
                st.info("No readable text found. Trying OCR (scanned PDF?)...")
                raw_text = extract_text_with_ocr(TEMP_PATH)

    if not raw_text or "Error" in raw_text:
        st.error("❌ Could not extract text from the PDF.")
    else:
        st.success("✅ Text extracted successfully!")
        
        # Truncate if needed
        truncated_text = raw_text[:max_chars]
        token_estimate = len(truncated_text) // 4  # rough estimate

        st.write(f"📝 **Preview of Extracted Text (first {len(truncated_text)} chars, ~{token_estimate} tokens):**")
        with st.expander("📄 View Extracted Text"):
            st.text_area("", value=truncated_text, height=200)

        # Summarize button
        if st.button("🧠 Summarize with Qwen3"):
            prompt = f"""
            Please provide a clear and concise summary of the following document. 
            Highlight key points, main ideas, and any conclusions. 
            If it's technical, legal, or academic, preserve important terms.

            Document:
            {truncated_text}

            Summary:
            """

            st.subheader("💬 Summary (Streaming from Qwen3...)")
            summary_container = st.empty()
            full_summary = ""

            # Use selected model
            for chunk in summarize_text_stream(prompt, model=model_choice):
                full_summary += chunk
                # Simulate real-time typing
                summary_container.markdown(full_summary + " <span style='opacity:0.5'>▌</span>", unsafe_allow_html=True)
                time.sleep(0.01)  # Smooth appearance

            # Final output without cursor
            summary_container.markdown(full_summary)