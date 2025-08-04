# Ollama Code Assistant (Enhanced Version)
import streamlit as st
import requests
import json
import os
from datetime import datetime
import time
from typing import Dict, List, Tuple, Optional
import base64
from pathlib import Path

# -------------------------------
# 🎨 Configuration & Styling
# -------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1:8b")
OLLAMA_API_URL = "http://localhost:11434/api/tags"

# Custom CSS for modern design
def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Header */
    .custom-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .custom-header h1 {
        margin: 0;
        font-weight: 700;
        font-size: 2.5rem;
    }
    
    .custom-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Status Cards */
    .status-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    
    .status-online {
        border-color: #10B981;
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
    }
    
    .status-offline {
        border-color: #EF4444;
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
    }
    
    /* Tool Cards */
    .tool-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #E5E7EB;
        transition: all 0.3s ease;
        cursor: pointer;
        margin: 0.5rem 0;
    }
    
    .tool-card:hover {
        border-color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }
    
    .tool-card.active {
        border-color: #667eea;
        background: linear-gradient(135deg, #F3F4F6 0%, #E5E7EB 100%);
    }
    
    /* Code Editor */
    .code-editor {
        border: 2px solid #E5E7EB;
        border-radius: 8px;
        background: #F9FAFB;
    }
    
    /* Buttons */
    .custom-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .custom-button:hover {
        transform: translateY(-1px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Progress Bar */
    .progress-container {
        background: #E5E7EB;
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #667eea, #764ba2);
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    /* History Panel */
    .history-item {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #667eea;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .history-item:hover {
        background: #F9FAFB;
        transform: translateX(5px);
    }
    
    /* Templates */
    .template-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .template-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .template-card:hover {
        border-color: #667eea;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .custom-header h1 {
            font-size: 2rem;
        }
        
        .template-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# 🗄️ Session State Management
# -------------------------------
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'canvas_content': '',
        'selected_mode': 'generate',
        'last_result': '',
        'ollama_alive': False,
        'request_history': [],
        'favorites': [],
        'current_template': None,
        'api_response_time': 0,
        'model_info': {},
        'selected_model': MODEL_NAME,
        'settings': {
            'auto_save': True,
            'syntax_highlight': True,
            'dark_mode': False,
            'max_history': 50
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# -------------------------------
# 🔗 Enhanced Ollama Functions
# -------------------------------
class OllamaManager:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.api_url = f"{self.base_url}/api/generate"
        self.tags_url = f"{self.base_url}/api/tags"
        
    def check_status(self) -> Tuple[bool, Dict]:
        """Enhanced status check with model information"""
        try:
            response = requests.get(self.tags_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                return True, {
                    'status': 'online',
                    'models': models,
                    'model_count': len(models),
                    'available_models': [m.get('name', '') for m in models]
                }
        except Exception as e:
            return False, {'status': 'offline', 'error': str(e)}
        
        return False, {'status': 'offline', 'error': 'Unknown error'}
    
    def send_request(self, prompt: str, model: str = None) -> Tuple[bool, str, float]:
        """Enhanced request with timing and better error handling"""
        if model is None:
            model = st.session_state.get('selected_model', MODEL_NAME)
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048
            }
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                try:
                    result = response.json().get("response", "").strip()
                    return True, result, response_time
                except json.JSONDecodeError:
                    return False, "❌ Failed to parse JSON response", response_time
            else:
                return False, f"❌ HTTP {response.status_code}: {response.text}", response_time
                
        except requests.exceptions.ConnectionError:
            return False, "❌ Cannot connect to Ollama. Is `ollama serve` running?", 0
        except requests.exceptions.Timeout:
            return False, "❌ ⏰ Request timed out (60s)", 0
        except Exception as e:
            return False, f"❌ Unexpected error: {str(e)}", 0

ollama = OllamaManager()

# -------------------------------
# 📝 Templates System
# -------------------------------
TEMPLATES = {
    "Python Script": {
        "icon": "🐍",
        "description": "Basic Python script template",
        "content": "# Python Script Template\n\ndef main():\n    \"\"\"\n    Main function\n    \"\"\"\n    pass\n\nif __name__ == \"__main__\":\n    main()",
        "language": "python"
    },
    "Flask API": {
        "icon": "🌐",
        "description": "Flask REST API template",
        "content": "from flask import Flask, jsonify, request\n\napp = Flask(__name__)\n\n@app.route('/api/health', methods=['GET'])\ndef health_check():\n    return jsonify({'status': 'healthy'})\n\nif __name__ == '__main__':\n    app.run(debug=True)",
        "language": "python"
    },
    "SQL Query": {
        "icon": "📊",
        "description": "SQL query template",
        "content": "-- SQL Query Template\nSELECT \n    column1,\n    column2\nFROM \n    table_name\nWHERE \n    condition = 'value'\nORDER BY \n    column1;",
        "language": "sql"
    },
    "React Component": {
        "icon": "⚛️",
        "description": "React functional component",
        "content": "import React, { useState } from 'react';\n\nconst ComponentName = () => {\n    const [state, setState] = useState('');\n    \n    return (\n        <div>\n            <h1>Component Title</h1>\n        </div>\n    );\n};\n\nexport default ComponentName;",
        "language": "javascript"
    }
}

# -------------------------------
# 🛠️ Enhanced Tool Functions
# -------------------------------
def save_to_history(mode: str, prompt: str, result: str, response_time: float):
    """Save request to history with metadata"""
    history_item = {
        'timestamp': datetime.now().isoformat(),
        'mode': mode,
        'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt,
        'result': result,
        'response_time': response_time,
        'favorite': False
    }
    
    st.session_state.request_history.insert(0, history_item)
    
    # Limit history size
    max_history = st.session_state.settings.get('max_history', 50)
    if len(st.session_state.request_history) > max_history:
        st.session_state.request_history = st.session_state.request_history[:max_history]

def export_content(content: str, format_type: str = "txt") -> bytes:
    """Export content in different formats"""
    if format_type == "txt":
        return content.encode('utf-8')
    elif format_type == "json":
        data = {
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "tool": "Ollama Code Assistant",
                "version": "2.0"
            }
        }
        return json.dumps(data, indent=2).encode('utf-8')
    elif format_type == "html":
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Code Export</title>
            <style>
                body {{ font-family: 'Courier New', monospace; margin: 40px; }}
                pre {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Code Export</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <pre><code>{content}</code></pre>
        </body>
        </html>
        """
        return html_content.encode('utf-8')

# -------------------------------
# 🎨 UI Components
# -------------------------------
def render_header():
    """Render custom header"""
    st.markdown("""
    <div class="custom-header">
        <h1>🧠 Ollama Code Assistant</h1>
        <p>AI-Powered Development Toolkit - Enhanced Edition</p>
    </div>
    """, unsafe_allow_html=True)

def render_status_card():
    """Render enhanced status card"""
    is_online, info = ollama.check_status()
    st.session_state.ollama_alive = is_online
    st.session_state.model_info = info
    
    if is_online:
        current_model = st.session_state.get('selected_model', MODEL_NAME)
        st.markdown(f"""
        <div class="status-card status-online">
            <h3>🟢 Ollama Status: Online</h3>
            <p><strong>Available Models:</strong> {info.get('model_count', 0)}</p>
            <p><strong>Current Model:</strong> {current_model}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-card status-offline">
            <h3>🔴 Ollama Status: Offline</h3>
            <p>Please start Ollama service: <code>ollama serve</code></p>
            <p><strong>Error:</strong> {info.get('error', 'Unknown')}</p>
        </div>
        """, unsafe_allow_html=True)

def render_mode_selector():
    """Render enhanced mode selector with cards"""
    mode_options = {
        "generate": {"icon": "✨", "title": "Generate Code", "desc": "Create new code from description"},
        "debug": {"icon": "🐞", "title": "Debug Code", "desc": "Find and fix code issues"},
        "auto-complete": {"icon": "⌨️", "title": "Auto-complete", "desc": "Complete partial code"},
        "sql-generator": {"icon": "📊", "title": "SQL Generator", "desc": "Generate SQL queries"},
        "readme-generator": {"icon": "📝", "title": "README Generator", "desc": "Create project documentation"},
        "code-documentation": {"icon": "📖", "title": "Document Code", "desc": "Add documentation to code"},
        "test-api": {"icon": "🧪", "title": "Test API", "desc": "Test API endpoints"}
    }
    
    st.markdown("### 🛠️ Select Tool")
    
    cols = st.columns(4)
    for i, (key, info) in enumerate(mode_options.items()):
        with cols[i % 4]:
            if st.button(
                f"{info['icon']} {info['title']}", 
                key=f"mode_{key}",
                help=info['desc'],
                use_container_width=True
            ):
                st.session_state.selected_mode = key
                st.rerun()

def render_templates_panel():
    """Render templates panel"""
    with st.expander("📋 Code Templates", expanded=False):
        st.markdown("Quick start with predefined templates:")
        
        cols = st.columns(2)
        for i, (name, template) in enumerate(TEMPLATES.items()):
            with cols[i % 2]:
                if st.button(
                    f"{template['icon']} {name}",
                    key=f"template_{name}",
                    help=template['description'],
                    use_container_width=True
                ):
                    st.session_state.canvas_content = template['content']
                    st.session_state.current_template = name
                    st.success(f"✅ Template '{name}' loaded!")
                    st.rerun()

def render_history_panel():
    """Render history panel"""
    if st.session_state.request_history:
        with st.expander(f"📚 History ({len(st.session_state.request_history)} items)", expanded=False):
            for i, item in enumerate(st.session_state.request_history[:10]):  # Show last 10
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    timestamp = datetime.fromisoformat(item['timestamp']).strftime('%H:%M:%S')
                    st.write(f"**{timestamp}** - {item['mode']}: {item['prompt']}")
                
                with col2:
                    if st.button("📋", key=f"load_history_{i}", help="Load to canvas"):
                        st.session_state.canvas_content = item['result']
                        st.rerun()
                
                with col3:
                    fav_icon = "⭐" if item.get('favorite') else "☆"
                    if st.button(fav_icon, key=f"fav_history_{i}", help="Toggle favorite"):
                        st.session_state.request_history[i]['favorite'] = not item.get('favorite', False)
                        st.rerun()

# -------------------------------
# 🎯 Main Application
# -------------------------------
def main():
    st.set_page_config(
        page_title="🧠 Ollama Code Assistant",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🧠"
    )
    
    # Load custom CSS
    load_css()
    
    # Initialize session state
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Model selection
        if st.session_state.model_info.get('available_models'):
            current_model = st.session_state.selected_model
            selected_model = st.selectbox(
                "Model",
                st.session_state.model_info['available_models'],
                index=0 if current_model not in st.session_state.model_info['available_models'] 
                      else st.session_state.model_info['available_models'].index(current_model)
            )
            if selected_model != current_model:
                st.session_state.selected_model = selected_model
        
        # URL configuration
        ollama_url = st.text_input("Ollama URL", value=OLLAMA_URL)
        
        st.markdown("---")
        
        # Settings
        st.markdown("### 🎛️ Preferences")
        st.session_state.settings['auto_save'] = st.checkbox("Auto-save results", value=st.session_state.settings['auto_save'])
        st.session_state.settings['syntax_highlight'] = st.checkbox("Syntax highlighting", value=st.session_state.settings['syntax_highlight'])
        st.session_state.settings['max_history'] = st.slider("Max history items", 10, 100, st.session_state.settings['max_history'])
        
        st.markdown("---")
        
        # Actions
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.request_history = []
            st.success("History cleared!")
        
        if st.button("💾 Export Settings", use_container_width=True):
            settings_json = json.dumps(st.session_state.settings, indent=2)
            st.download_button(
                "Download settings.json",
                settings_json,
                "ollama_settings.json",
                "application/json"
            )
    
    # Main content
    render_header()
    render_status_card()
    
    # Templates and History
    col1, col2 = st.columns(2)
    with col1:
        render_templates_panel()
    with col2:
        render_history_panel()
    
    # Mode selection
    render_mode_selector()
    
    # Main form
    mode = st.session_state.selected_mode
    
    with st.form("enhanced_form", clear_on_submit=False):
        st.markdown(f"### {mode.replace('-', ' ').title()}")
        
        # Dynamic input based on mode
        if mode == "generate":
            prompt = st.text_area("Describe what you want to generate:", height=150, placeholder="Example: Create a Python function to calculate fibonacci numbers")
            language = st.selectbox("Target Language", ["Python", "JavaScript", "Java", "C++", "Go", "Rust"])
            
        elif mode == "debug":
            prompt = st.text_area("Paste your code to debug:", height=200, placeholder="Paste your code here...")
            error_description = st.text_input("Describe the error (optional):", placeholder="What error are you encountering?")
            
        elif mode == "auto-complete":
            prompt = st.text_area("Paste incomplete code:", height=150, placeholder="Paste your partial code here...")
            language = st.selectbox("Language", ["Python", "JavaScript", "Java", "C++", "TypeScript"])
            
        elif mode == "sql-generator":
            prompt = st.text_area("Describe your query in natural language:", height=100, placeholder="Find all users who registered last month")
            database_type = st.selectbox("Database Type", ["MySQL", "PostgreSQL", "SQLite", "MongoDB"])
            schema = st.text_area("Database Schema:", height=100, placeholder="users(id, name, email, created_at)")
            
        elif mode == "readme-generator":
            prompt = st.text_area("Describe your project:", height=150, placeholder="A web application for managing tasks...")
            project_type = st.selectbox("Project Type", ["Web App", "Mobile App", "Desktop App", "Library", "API"])
            
        elif mode == "code-documentation":
            prompt = st.text_area("Paste code to document:", height=200, placeholder="Paste your code here...")
            doc_style = st.selectbox("Documentation Style", ["Google", "NumPy", "Sphinx", "JSDoc"])
            
        elif mode == "test-api":
            prompt = st.text_input("API URL:", placeholder="https://api.example.com/users")
            method = st.selectbox("HTTP Method", ["GET", "POST", "PUT", "DELETE"])
            headers = st.text_area("Headers (JSON):", value="{}", height=80)
            payload = st.text_area("Payload (JSON):", value="{}", height=80)
            expected_fields = st.text_input("Expected Response Fields:", placeholder="id, name, email")
        
        # Submit button
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            submitted = st.form_submit_button("🚀 Execute", use_container_width=True)
        with col3:
            clear_form = st.form_submit_button("🗑️ Clear", use_container_width=True)
    
    # Process submission
    if submitted and st.session_state.ollama_alive:
        if not prompt.strip():
            st.warning("⚠️ Please enter a prompt.")
        else:
            with st.spinner("🧠 AI is thinking..."):
                # Construct full prompt based on mode
                if mode == "generate":
                    full_prompt = f"Generate a clean, well-documented {language} solution for: {prompt}\n\nProvide complete, working code with comments."
                    
                elif mode == "debug":
                    full_prompt = f"Debug and fix this code:\n```\n{prompt}\n```\n\nError description: {error_description if 'error_description' in locals() else 'Not specified'}\n\nProvide the corrected code with explanations."
                    
                elif mode == "auto-complete":
                    full_prompt = f"Complete this {language} code:\n```{language.lower()}\n{prompt}\n```\n\nProvide the completed code with proper structure."
                    
                elif mode == "sql-generator":
                    full_prompt = f"Convert this natural language query to {database_type} SQL:\nQuery: {prompt}\nSchema: {schema if 'schema' in locals() else 'Not specified'}\n\nProvide optimized SQL query with comments."
                    
                elif mode == "readme-generator":
                    full_prompt = f"Generate a comprehensive README.md for a {project_type if 'project_type' in locals() else 'software project'}:\n{prompt}\n\nInclude: Description, Installation, Usage, Examples, Contributing guidelines."
                    
                elif mode == "code-documentation":
                    full_prompt = f"Add comprehensive documentation to this code using {doc_style if 'doc_style' in locals() else 'standard'} style:\n```\n{prompt}\n```\n\nInclude docstrings, inline comments, and type hints where applicable."
                    
                elif mode == "test-api":
                    # Handle API testing differently
                    try:
                        import requests
                        headers_dict = json.loads(headers) if headers.strip() else {}
                        payload_dict = json.loads(payload) if payload.strip() else {}
                        
                        start_time = time.time()
                        if method == "GET":
                            response = requests.get(prompt, headers=headers_dict, timeout=10)
                        elif method == "POST":
                            response = requests.post(prompt, json=payload_dict, headers=headers_dict, timeout=10)
                        elif method == "PUT":
                            response = requests.put(prompt, json=payload_dict, headers=headers_dict, timeout=10)
                        elif method == "DELETE":
                            response = requests.delete(prompt, headers=headers_dict, timeout=10)
                        
                        response_time = time.time() - start_time
                        
                        # Format response
                        result = f"""✅ API Test Results
**URL:** {prompt}
**Method:** {method}
**Status Code:** {response.status_code}
**Response Time:** {response_time:.2f}s

**Headers:**
{json.dumps(dict(response.headers), indent=2)}

**Response Body:**
```json
{json.dumps(response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text, indent=2)}
```
"""
                        
                        st.session_state.canvas_content = result
                        st.session_state.last_result = f"✅ API test completed in {response_time:.2f}s"
                        save_to_history(mode, prompt, result, response_time)
                        st.success(st.session_state.last_result)
                        
                    except Exception as e:
                        error_msg = f"❌ API test failed: {str(e)}"
                        st.error(error_msg)
                        st.session_state.last_result = error_msg
                    
                    st.rerun()
                    return
                
                # For non-API modes, use Ollama
                success, result, response_time = ollama.send_request(full_prompt)
                st.session_state.api_response_time = response_time
                
                if success:
                    st.session_state.canvas_content = result
                    st.session_state.last_result = f"✅ {mode.replace('-', ' ').title()} completed in {response_time:.2f}s"
                    save_to_history(mode, prompt, result, response_time)
                    st.success(st.session_state.last_result)
                else:
                    st.error(result)
                    st.session_state.last_result = f"❌ {result[:100]}..."
            
            st.rerun()
    
    elif submitted and not st.session_state.ollama_alive:
        st.error("❌ Ollama is not available. Please check the connection.")
    
    # Clear form
    if clear_form:
        st.session_state.canvas_content = ""
        st.session_state.last_result = ""
        st.rerun()
    
    # Enhanced Canvas Panel
    st.markdown("---")
    st.markdown("### 📝 Code Canvas")
    
    # Canvas tabs
    tab1, tab2, tab3 = st.tabs(["✏️ Editor", "👁️ Preview", "📤 Export"])
    
    with tab1:
        # Editor with enhanced features
        col1, col2 = st.columns([4, 1])
        
        with col1:
            canvas_content = st.text_area(
                "Edit your code:",
                value=st.session_state.canvas_content,
                height=400,
                key="canvas_editor",
                help="Use Ctrl+A to select all, Ctrl+C to copy"
            )
            
            if canvas_content != st.session_state.canvas_content:
                st.session_state.canvas_content = canvas_content
                if st.session_state.settings['auto_save']:
                    st.session_state.last_result = "💾 Auto-saved"
        
        with col2:
            st.markdown("**Quick Actions:**")
            
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.code(st.session_state.canvas_content)  # This makes it easier to copy
                st.success("Select and copy the code above!")
            
            if st.button("🔄 Format Code", use_container_width=True):
                # Basic formatting (could be enhanced with actual formatters)
                formatted = st.session_state.canvas_content.strip()
                st.session_state.canvas_content = formatted
                st.rerun()
            
            if st.button("📏 Count Lines", use_container_width=True):
                lines = len(st.session_state.canvas_content.split('\n'))
                chars = len(st.session_state.canvas_content)
                st.info(f"📊 {lines} lines, {chars} characters")
    
    with tab2:
        # Enhanced preview with syntax detection
        content = st.session_state.canvas_content
        
        if content.strip():
            # Auto-detect language
            language = "text"
            if "def " in content or "import " in content: language = "python"
            elif "function" in content or "const " in content: language = "javascript"
            elif "SELECT" in content.upper(): language = "sql"
            elif "<html" in content.lower(): language = "html"
            elif "{" in content and ":" in content: language = "json"
            elif "class " in content and "public" in content: language = "java"
            
            col1, col2 = st.columns([3, 1])
            with col2:
                st.info(f"🏷️ Detected: {language.title()}")
            
            st.code(content, language=language)
            
            # Code statistics
            lines = len(content.split('\n'))
            words = len(content.split())
            chars = len(content)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Lines", lines)
            with col2:
                st.metric("Words", words)
            with col3:
                st.metric("Characters", chars)
        else:
            st.info("👈 Generate or write some code to see the preview!")
    
    with tab3:
        # Enhanced export options
        if st.session_state.canvas_content.strip():
            st.markdown("**Export Options:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    "💾 Download as TXT",
                    data=export_content(st.session_state.canvas_content, "txt"),
                    file_name=f"code_{int(datetime.now().timestamp())}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                st.download_button(
                    "📄 Download as JSON",
                    data=export_content(st.session_state.canvas_content, "json"),
                    file_name=f"export_{int(datetime.now().timestamp())}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col3:
                st.download_button(
                    "🌐 Download as HTML",
                    data=export_content(st.session_state.canvas_content, "html"),
                    file_name=f"code_{int(datetime.now().timestamp())}.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            # Share options
            st.markdown("**Share:**")
            share_url = f"data:text/plain;base64,{base64.b64encode(st.session_state.canvas_content.encode()).decode()}"
            st.markdown(f"📎 **Share URL:** [Copy this link]({share_url})")
            
        else:
            st.info("📝 No content to export. Generate some code first!")
    
    # Performance metrics
    if st.session_state.api_response_time > 0:
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("⏱️ Last Response Time", f"{st.session_state.api_response_time:.2f}s")
        with col2:
            st.metric("📚 History Items", len(st.session_state.request_history))
        with col3:
            st.metric("⭐ Favorites", sum(1 for item in st.session_state.request_history if item.get('favorite')))
        with col4:
            st.metric("🎯 Current Mode", st.session_state.selected_mode.title())
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🧠 <strong>Ollama Code Assistant v2.0</strong> | Powered by <a href="https://ollama.com/" target="_blank">Ollama</a> | 
        Built with ❤️ using <a href="https://streamlit.io/" target="_blank">Streamlit</a></p>
        <p><em>Local AI • Privacy First • Developer Focused</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()