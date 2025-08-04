# AI Applications Hub - Main Launcher
import streamlit as st
import requests
import subprocess
import sys
import os
from datetime import datetime
import json

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_API_URL = "http://localhost:11434/api/tags"

# Applications Configuration
APPLICATIONS = {
    "🔎 Sentiment Analysis": {
        "file": "app_sentimet_analysis.py",
        "description": "Analyze sentiment and identify contributing words",
        "model": "Mistral-Nemo",
        "category": "Text Analysis",
        "features": ["Multi-language", "Word highlighting", "Real-time analysis"]
    },
    "🤖 AI Virtual Assistant": {
        "file": "app_assistai.py", 
        "description": "Conversational assistant with task scheduling",
        "model": "Llama3.1:8b",
        "category": "Productivity",
        "features": ["Task scheduling", "Conversation", "Reminders"]
    },
    "🛒 E-commerce Recommendations": {
        "file": "app_ecommerce.py",
        "description": "Product recommendations based on preferences", 
        "model": "Granite3.2",
        "category": "Business",
        "features": ["Product matching", "Preference analysis", "Catalog integration"]
    },
    "📧 Email Responder": {
        "file": "app_email_responder.py",
        "description": "Generate professional email responses",
        "model": "Mistral-Nemo", 
        "category": "Communication",
        "features": ["Auto language detection", "Multiple tones", "Professional formatting"]
    },
    "⚖️ Legal Assistant": {
        "file": "app_legal_assistant.py",
        "description": "Generate professional legal contracts",
        "model": "Mistral-Nemo",
        "category": "Legal",
        "features": ["Multiple contract types", "Multi-language", "PDF/DOCX export"]
    },
    "🩺 Medical Symptom Analyzer": {
        "file": "app_medical_assist.py",
        "description": "Analyze symptoms and provide general advice",
        "model": "MedLLaMA2:7b",
        "category": "Healthcare",
        "features": ["Symptom analysis", "General advice", "Medical terminology"]
    },
    "📝 Meeting Minutes Generator": {
        "file": "app_meeting.py",
        "description": "Audio transcription and meeting minutes",
        "model": "Mistral-Nemo",
        "category": "Productivity",
        "features": ["Audio transcription", "Participant detection", "Professional export"]
    },
    "🧠 Named Entity Recognition": {
        "file": "app_ner.py",
        "description": "Extract entities from text",
        "model": "Multiple Models",
        "category": "Text Analysis", 
        "features": ["Multi-model support", "Entity extraction", "Structured output"]
    },
    "📄 PDF Summarizer": {
        "file": "app_pdf_text_extractor.py",
        "description": "Extract and summarize PDF content",
        "model": "Qwen3",
        "category": "Document Processing",
        "features": ["OCR support", "Text extraction", "Intelligent summarization"]
    },
    "🚀 Resume Generator": {
        "file": "app_resume_generator.py", 
        "description": "Generate ATS-optimized professional resumes",
        "model": "Mistral-Nemo",
        "category": "Career",
        "features": ["ATS optimization", "Multi-language", "Professional export"]
    },
    "💬 Support Chatbot": {
        "file": "app_support_chatboot.py",
        "description": "Intelligent customer support chatbot", 
        "model": "Various",
        "category": "Customer Service",
        "features": ["Customer support", "Multi-turn conversation", "Context awareness"]
    },
    "🎤 Voice Lab": {
        "file": "app_voice_lab.py",
        "description": "Advanced voice processing interface",
        "model": "Various", 
        "category": "Audio Processing",
        "features": ["Voice processing", "Audio analysis", "Speech recognition"]
    }
}

def check_ollama_status():
    """Check if Ollama service is running and available"""
    try:
        response = requests.get(OLLAMA_API_URL, timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return True, len(models), [model.get('name', '') for model in models]
        return False, 0, []
    except:
        return False, 0, []

def get_system_info():
    """Get system information"""
    return {
        "Python Version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "Platform": sys.platform,
        "Streamlit Version": st.__version__,
        "Current Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def launch_application(app_file):
    """Launch a specific application"""
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
    # Page configuration
    st.set_page_config(
        page_title="AI Applications Hub",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
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
    .status-good { color: #28a745; font-weight: bold; }
    .status-bad { color: #dc3545; font-weight: bold; }
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

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Applications Hub</h1>
        <p>Centralized launcher for AI-powered tools and assistants</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar - System Status
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Ollama Status
        is_running, model_count, models = check_ollama_status()
        
        if is_running:
            st.markdown('<p class="status-good">✅ Ollama: Connected</p>', unsafe_allow_html=True)
            st.metric("Available Models", model_count)
            
            with st.expander("📋 Installed Models"):
                for model in models:
                    st.text(f"• {model}")
        else:
            st.markdown('<p class="status-bad">❌ Ollama: Disconnected</p>', unsafe_allow_html=True)
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
        
        # System Information
        st.header("ℹ️ System Info")
        system_info = get_system_info()
        for key, value in system_info.items():
            st.text(f"{key}: {value}")

        st.divider()
        
        # Quick Actions
        st.header("⚡ Quick Actions")
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()
            
        if st.button("📊 Show All Apps", use_container_width=True):
            st.session_state.show_all = True

    # Main Content
    st.header("🚀 Available Applications")
    
    # Category filter
    categories = list(set([app["category"] for app in APPLICATIONS.values()]))
    selected_category = st.selectbox("Filter by Category", ["All"] + sorted(categories))

    # Search
    search_term = st.text_input("🔍 Search Applications", placeholder="Type to search...")

    # Application Grid
    filtered_apps = APPLICATIONS.items()
    
    if selected_category != "All":
        filtered_apps = [(name, info) for name, info in filtered_apps if info["category"] == selected_category]
    
    if search_term:
        filtered_apps = [(name, info) for name, info in filtered_apps 
                        if search_term.lower() in name.lower() or search_term.lower() in info["description"].lower()]

    # Display applications in columns
    col1, col2 = st.columns(2)
    
    for i, (app_name, app_info) in enumerate(filtered_apps):
        col = col1 if i % 2 == 0 else col2
        
        with col:
            with st.container():
                st.markdown(f"""
                <div class="app-card">
                    <h3>{app_name}</h3>
                    <p><strong>Model:</strong> {app_info['model']}</p>
                    <p><strong>Category:</strong> {app_info['category']}</p>
                    <p>{app_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Features
                st.markdown("**Features:**")
                features_html = ""
                for feature in app_info['features']:
                    features_html += f'<span class="feature-tag">{feature}</span>'
                st.markdown(features_html, unsafe_allow_html=True)
                
                # Launch button
                col_launch, col_info = st.columns([2, 1])
                
                with col_launch:
                    if st.button(f"🚀 Launch", key=f"launch_{app_info['file']}", use_container_width=True):
                        if launch_application(app_info['file']):
                            st.success(f"Launching {app_name}...")
                        
                with col_info:
                    if st.button("ℹ️", key=f"info_{app_info['file']}", help="Show details"):
                        st.session_state[f"show_info_{app_info['file']}"] = True
                
                # Show detailed info if requested
                if st.session_state.get(f"show_info_{app_info['file']}", False):
                    with st.expander(f"📋 {app_name} Details", expanded=True):
                        st.markdown(f"""
                        **File:** `{app_info['file']}`  
                        **AI Model:** {app_info['model']}  
                        **Category:** {app_info['category']}  
                        **Description:** {app_info['description']}
                        
                        **Features:**
                        """)
                        for feature in app_info['features']:
                            st.markdown(f"• {feature}")
                        
                        if st.button("❌ Close", key=f"close_{app_info['file']}"):
                            st.session_state[f"show_info_{app_info['file']}"] = False
                            st.rerun()
                
                st.divider()

    # Statistics
    st.header("📊 Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Applications", len(APPLICATIONS))
    col2.metric("Categories", len(categories))
    col3.metric("AI Models Used", len(set([app["model"] for app in APPLICATIONS.values()])))
    col4.metric("Ollama Models", model_count if is_running else 0)

    # Instructions
    st.header("📖 Instructions")
    
    with st.expander("🚀 How to Use"):
        st.markdown("""
        ### Getting Started
        
        1. **Check System Status** in the sidebar
        2. **Ensure Ollama is running** (`ollama serve`)  
        3. **Install required models** (see sidebar for commands)
        4. **Browse applications** by category or search
        5. **Click "🚀 Launch"** to open any application
        
        ### Navigation Tips
        
        - Use the **category filter** to find specific types of applications
        - **Search by name or description** to quickly locate tools
        - Click **"ℹ️"** for detailed information about each application
        - Check the **sidebar** for system status and troubleshooting
        
        ### Troubleshooting
        
        - **Ollama not connected?** Run `ollama serve` in terminal
        - **Missing models?** Install with `ollama pull model-name`
        - **Application won't launch?** Check if the file exists in the directory
        - **Performance issues?** Ensure sufficient RAM for AI models
        """)

    with st.expander("🔧 Technical Details"):
        st.markdown(f"""
        ### System Configuration
        
        **Applications Directory:** `{os.getcwd()}`  
        **Python Executable:** `{sys.executable}`  
        **Ollama API:** `{OLLAMA_API_URL}`
        
        ### Available Models by Category
        """)
        
        model_usage = {}
        for app_info in APPLICATIONS.values():
            model = app_info['model']
            if model not in model_usage:
                model_usage[model] = []
            model_usage[model].append(app_info['file'])
        
        for model, files in model_usage.items():
            st.markdown(f"**{model}:**")
            for file in files:
                st.markdown(f"  • {file}")

    # Footer
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