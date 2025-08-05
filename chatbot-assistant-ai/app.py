# AI Applications Hub (Enhanced Auto-Discovery Version)
__version__ = "1.0.0"

import streamlit as st
import requests
import subprocess
import sys
import os
import glob
from datetime import datetime
import json

# Ollama config
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_API_URL = "http://localhost:11434/api/tags"

# Discover all app_*.py (except the hub itself)
def discover_apps():
    apps = {}
    py_files = sorted([f for f in glob.glob("app_*.py") if f != "app.py"])
    app_name_map = {
        "app_assistai.py": "🤖 AI Virtual Assistant",
        "app_ecommerce.py": "🛒 E-commerce Recommendations",
        "app_email_responder.py": "📧 Email Responder",
        "app_legal_assistant.py": "⚖️ Legal Assistant",
        "app_medical_assist.py": "🩺 Medical Symptom Analyzer",
        "app_meeting.py": "📝 Meeting Minutes Generator",
        "app_ner.py": "🧠 Named Entity Recognition",
        "app_pdf_text_extractor.py": "📄 PDF Summarizer",
        "app_resume_generator.py": "🚀 Resume Generator",
        "app_sentimet_analysis.py": "🔎 Sentiment Analysis",
        "app_support_chatboot.py": "💬 Support Chatbot",
        "app_voice_lab.py": "🎤 Voice Lab"
    }
    for file in py_files:
        app_display = app_name_map.get(file, file.replace("app_", "").replace(".py", "").replace("_", " ").title())
        apps[file] = {"display": app_display, "file": file}
    return apps

def check_ollama_status():
    try:
        response = requests.get(OLLAMA_API_URL, timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return True, len(models), [model.get('name', '') for model in models]
        return False, 0, []
    except:
        return False, 0, []

def get_system_info():
    return {
        "Project Version": __version__,
        "Python Version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "Platform": sys.platform,
        "Streamlit Version": st.__version__,
        "Current Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def launch_application(app_file):
    if os.path.exists(app_file):
        try:
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", app_file])
            return True
        except Exception as e:
            st.error(f"Failed to launch {app_file}: {str(e)}")
            return False
    else:
        st.error(f"Application file {app_file} not found!")
        return False

def main():
    st.set_page_config(
        page_title="AI Applications Hub",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .app-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .feature-tag {
        background: #e9ecef;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="main-header">
        <h1>🤖 AI Applications Hub <small style="font-size:1rem">v{__version__}</small></h1>
        <p>Centralized launcher for AI-powered tools — <b>auto-discovers new apps!</b></p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("🔧 System Status")
        is_running, model_count, models = check_ollama_status()
        if is_running:
            st.success('✅ Ollama: Connected')
            st.metric("Available Models", model_count)
            with st.expander("📋 Installed Models"):
                for model in models:
                    st.text(f"• {model}")
        else:
            st.error('❌ Ollama: Disconnected')
            with st.expander("🔧 Setup Instructions"):
                st.markdown("""
                **Start Ollama:**
                ```bash
                ollama serve
                ```
                **Install Models:**
                ```bash
                ollama pull mistral-nemo:latest
                ollama pull llama3.1:8b
                ollama pull qwen3:14b
                ```
                """)
        st.divider()
        st.header("ℹ️ System Info")
        for key, value in get_system_info().items():
            st.text(f"{key}: {value}")
        st.divider()
        st.header("📖 Documentation")
        st.markdown("[View README](./README.md)", unsafe_allow_html=True)
        st.divider()
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()

    st.header("🚀 Available Applications")
    apps = discover_apps()
    col1, col2 = st.columns(2)
    for i, (app_file, app_meta) in enumerate(apps.items()):
        col = col1 if i % 2 == 0 else col2
        with col:
            with st.container():
                st.markdown(f"""
                <div class="app-card">
                    <h3>{app_meta['display']}</h3>
                    <p><strong>File:</strong> <code>{app_file}</code></p>
                </div>
                """, unsafe_allow_html=True)
                col_launch, col_info = st.columns([2, 1])
                with col_launch:
                    if st.button(f"🚀 Launch", key=f"launch_{app_file}", use_container_width=True):
                        if launch_application(app_file):
                            st.success(f"Launching {app_meta['display']} ...")
                with col_info:
                    if st.button("ℹ️", key=f"info_{app_file}"):
                        st.session_state[f"show_info_{app_file}"] = True
                if st.session_state.get(f"show_info_{app_file}", False):
                    with st.expander(f"📋 {app_meta['display']} Details", expanded=True):
                        st.markdown(f"""
                        **File:** `{app_file}`  
                        **Name:** {app_meta['display']}
                        """)
                        if st.button("❌ Close", key=f"close_{app_file}"):
                            st.session_state[f"show_info_{app_file}"] = False
                            st.rerun()
                st.divider()
    st.header("📊 Statistics")
    st.metric("Total Applications", len(apps))
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px;'>
        <h4>🤖 AI Applications Hub</h4>
        <p>Powered by Ollama | Built with Streamlit | Open Source</p>
        <p style='color: #666; font-size: 0.9em;'>
            Launch any AI application from this central hub. Ensure Ollama is running for full functionality.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()