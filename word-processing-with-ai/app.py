# AI Word Processing Suite - Main Dashboard
# AI Word Processing Suite - Enhanced Main Dashboard
import streamlit as st
import requests
import json
from datetime import datetime
import time
import os
import io
from io import StringIO
from dotenv import load_dotenv
import re
import hashlib
from typing import Dict, List, Optional, Tuple

# Load environment variables
load_dotenv()

# Try to import optional dependencies with better error handling
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_TAGS_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").replace("/api/generate", "") + "/api/tags"
NEWS_API_URL = "https://newsapi.org/v2/top-headlines"
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Available models based on your ollama list
AVAILABLE_MODELS = {
    "mistral:latest": "Mistral Latest - Fast and efficient for summarization",
    "llama3.1:8b": "Llama 3.1 8B - Excellent general purpose model",
    "codellama:13b": "CodeLlama 13B - Great for technical content analysis",
    "deepseek-r1:latest": "DeepSeek R1 Latest - Advanced reasoning capabilities",
    "deepseek-r1:1.5b": "DeepSeek R1 1.5B - Lightweight but capable",
    "phi4:latest": "Phi4 Latest - Microsoft's efficient model",
    "gemma3:12b": "Gemma3 12B - Google's powerful text model",
    "gemma3n:e4b": "Gemma3N E4B - Specialized Gemma variant"
}

# Page configuration
st.set_page_config(
    page_title="AI Word Processing Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS for the dashboard
st.markdown("""
<style>
/* Global styles */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.app-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    margin: 1rem 0;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    text-decoration: none;
}

.app-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    text-decoration: none;
    color: white;
}

.app-title {
    font-size: 1.5rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
}

.app-description {
    font-size: 1rem;
    opacity: 0.9;
    line-height: 1.4;
}

.dashboard-header {
    text-align: center;
    padding: 3rem 0;
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}

.dashboard-header h1 {
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 1rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.dashboard-header p {
    font-size: 1.3rem;
    opacity: 0.9;
    margin: 0;
}

.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse 2s infinite;
}

.status-online {
    background-color: #4CAF50;
}

.status-offline {
    background-color: #f44336;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.news-card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #1f77b4;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}

.news-card:hover {
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
}

.summary-box {
    background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
    padding: 2rem;
    border-radius: 12px;
    border-left: 4px solid #4CAF50;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.feature-card {
    background: white;
    padding: 2rem;
    border-radius: 12px;
    border: 1px solid #e1e5e9;
    transition: all 0.3s ease;
    cursor: pointer;
}

.feature-card:hover {
    border-color: #667eea;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    transform: translateY(-2px);
}

.stats-container {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #e1e5e9;
    margin: 1rem 0;
}

.error-container {
    background: #fee;
    border: 1px solid #fcc;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    color: #c33;
}

.success-container {
    background: #efe;
    border: 1px solid #cfc;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    color: #363;
}

/* Responsive design */
@media (max-width: 768px) {
    .dashboard-header h1 {
        font-size: 2rem;
    }
    
    .feature-grid {
        grid-template-columns: 1fr;
    }
    
    .app-card {
        padding: 1.5rem;
    }
}

/* Button enhancements */
.stButton > button {
    border-radius: 8px;
    border: none;
    padding: 0.75rem 1.5rem;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

/* Sidebar enhancements */
.stSidebar {
    background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
}

/* Form enhancements */
.stTextArea textarea {
    border-radius: 8px;
    border: 2px solid #e1e5e9;
    transition: border-color 0.3s ease;
}

.stTextArea textarea:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}
</style>
""", unsafe_allow_html=True)

# Initialize session state with enhanced error handling
def init_session_state():
    """Initialize session state variables with defaults."""
    defaults = {
        'current_app': 'dashboard',
        'canvas_content': "",
        'news_data': None,
        'article_contents': {},
        'selected_model': list(AVAILABLE_MODELS.keys())[0],
        'ollama_status': 'unknown',
        'available_models': [],
        'processing_history': [],
        'user_preferences': {
            'auto_save': True,
            'show_advanced': False,
            'preferred_models': {}
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Enhanced Ollama connection check
def check_ollama_status() -> Tuple[bool, Dict]:
    """Check Ollama connection and get available models."""
    try:
        # Test connection with tags endpoint
        response = requests.get(OLLAMA_TAGS_URL, timeout=3)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            available_models = [model.get('name', '') for model in models]
            
            # Update session state
            st.session_state.ollama_status = 'online'
            st.session_state.available_models = available_models
            
            return True, {
                'status': 'online',
                'models': available_models,
                'count': len(available_models)
            }
    except requests.exceptions.RequestException as e:
        st.session_state.ollama_status = 'offline'
        return False, {
            'status': 'offline',
            'error': str(e)
        }
    
    return False, {'status': 'offline', 'error': 'Unknown error'}

# Enhanced Ollama request with better error handling
def make_ollama_request(prompt: str, model: str = None, temperature: float = 0.7) -> Tuple[bool, str, float]:
    """Make request to Ollama with enhanced error handling."""
    if model is None:
        model = st.session_state.get('selected_model', list(AVAILABLE_MODELS.keys())[0])
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": 2048
        }
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            try:
                result = response.json().get("response", "").strip()
                
                # Save to processing history
                if 'processing_history' not in st.session_state:
                    st.session_state.processing_history = []
                
                st.session_state.processing_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'model': model,
                    'prompt_length': len(prompt),
                    'response_time': response_time,
                    'success': True
                })
                
                return True, result, response_time
            except json.JSONDecodeError:
                return False, "❌ Failed to parse response from Ollama", response_time
        else:
            return False, f"❌ HTTP {response.status_code}: {response.text[:200]}", response_time
            
    except requests.exceptions.Timeout:
        return False, "⏰ Request timed out. Try with a shorter prompt or different model.", time.time() - start_time
    except requests.exceptions.ConnectionError:
        return False, "🔌 Cannot connect to Ollama. Ensure 'ollama serve' is running.", time.time() - start_time
    except Exception as e:
        return False, f"❌ Unexpected error: {str(e)}", time.time() - start_time

# Application definitions with enhanced metadata
APPLICATIONS = {
    'dashboard': {
        'title': 'Dashboard',
        'description': 'Main dashboard and application selector',
        'icon': '🏠',
        'category': 'navigation'
    },
    'news_summarizer': {
        'title': 'News Summarizer',
        'description': 'Fetch and summarize latest news articles with AI-powered insights',
        'icon': '📰',
        'model': 'mistral:latest',
        'category': 'content'
    },
    'code_generator': {
        'title': 'Code Generator & Debugger',
        'description': 'Generate, debug, and optimize code with AI assistance',
        'icon': '💻',
        'model': 'codellama:13b',
        'category': 'development'
    },
    'legal_analyzer': {
        'title': 'Legal Document Analyzer',
        'description': 'Analyze legal documents for key insights, risks, and obligations',
        'icon': '📄',
        'model': 'phi4:latest',
        'category': 'analysis'
    },
    'content_writer': {
        'title': 'Content Writer',
        'description': 'Generate blog posts, articles, and creative content with various styles',
        'icon': '✍️',
        'model': 'llama3.1:8b',
        'category': 'content'
    },
    'grammar_checker': {
        'title': 'Grammar & Spell Checker',
        'description': 'Proofread and correct grammar, spelling, and sentence structure',
        'icon': '📝',
        'model': 'mistral:latest',
        'category': 'editing'
    },
    'text_summarizer': {
        'title': 'Text Summarizer',
        'description': 'Create concise, intelligent summaries of long texts',
        'icon': '📋',
        'model': 'mistral:latest',
        'category': 'content'
    }
}

# Enhanced sidebar with better status display
def render_sidebar():
    """Render enhanced sidebar with status and navigation."""
    with st.sidebar:
        st.markdown("### 🚀 AI Word Processing Suite")
        
        # Enhanced Ollama status
        ollama_online, status_info = check_ollama_status()
        status_class = "status-online" if ollama_online else "status-offline"
        status_text = "Online" if ollama_online else "Offline"
        
        st.markdown(f"""
        <div style="margin: 1rem 0; padding: 1rem; background: white; border-radius: 8px; border-left: 4px solid {'#4CAF50' if ollama_online else '#f44336'};">
            <strong>Ollama Status:</strong><br>
            <span class="status-indicator {status_class}"></span>{status_text}
            {f"<br><small>{status_info.get('count', 0)} models available</small>" if ollama_online else f"<br><small>{status_info.get('error', 'Check connection')}</small>"}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation with categories
        st.markdown("### 📱 Applications")
        
        categories = {
            'content': '📄 Content Tools',
            'development': '💻 Development',
            'analysis': '🔍 Analysis',
            'editing': '✏️ Editing'
        }
        
        for category, apps in [(cat, [k for k, v in APPLICATIONS.items() if v.get('category') == cat]) for cat in categories]:
            if apps and category != 'navigation':
                st.markdown(f"**{categories[category]}**")
                for app_key in apps:
                    app_info = APPLICATIONS[app_key]
                    if st.button(f"{app_info['icon']} {app_info['title']}", 
                                use_container_width=True,
                                key=f"nav_{app_key}"):
                        st.session_state.current_app = app_key
                        st.rerun()
        
        # Dashboard button
        if st.button("🏠 Dashboard", use_container_width=True, key="nav_dashboard"):
            st.session_state.current_app = 'dashboard'
            st.rerun()
        
        st.markdown("---")
        
        # Enhanced model selection
        if st.session_state.current_app != 'dashboard':
            current_app_info = APPLICATIONS[st.session_state.current_app]
            if 'model' in current_app_info:
                st.markdown("### ⚙️ Model Selection")
                
                # Show recommended model
                recommended_model = current_app_info['model']
                st.caption(f"💡 Recommended: {recommended_model}")
                
                # Model selector
                available_for_selection = list(AVAILABLE_MODELS.keys())
                if st.session_state.available_models:
                    available_for_selection = [m for m in available_for_selection if m in st.session_state.available_models]
                
                if available_for_selection:
                    default_index = 0
                    if recommended_model in available_for_selection:
                        default_index = available_for_selection.index(recommended_model)
                    
                    selected_model = st.selectbox(
                        "Choose AI Model",
                        options=available_for_selection,
                        format_func=lambda x: AVAILABLE_MODELS.get(x, x),
                        index=default_index,
                        key="model_selector"
                    )
                    st.session_state.selected_model = selected_model
                else:
                    st.warning("⚠️ No models available. Please pull models in Ollama.")
        
        # Processing history
        if st.session_state.get('processing_history'):
            with st.expander("📊 Processing Stats"):
                history = st.session_state.processing_history[-10:]  # Last 10 requests
                avg_time = sum(h['response_time'] for h in history) / len(history)
                st.metric("Avg Response Time", f"{avg_time:.2f}s")
                st.metric("Total Requests", len(st.session_state.processing_history))

# Enhanced dashboard rendering
def render_dashboard():
    """Render enhanced main dashboard."""
    st.markdown("""
    <div class="dashboard-header">
        <h1>🤖 AI Word Processing Suite</h1>
        <p>Powerful AI-driven text processing tools powered by local Ollama models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System status overview
    ollama_online, status_info = check_ollama_status()
    
    if ollama_online:
        st.success(f"🟢 System Ready - {status_info.get('count', 0)} models available")
    else:
        st.error(f"🔴 System Offline - {status_info.get('error', 'Unknown error')}")
        st.info("💡 **Quick Fix:** Start Ollama with `ollama serve` and refresh this page.")
    
    # Feature categories
    st.markdown("## 🛠️ Available Tools")
    
    categories = {
        'content': {
            'title': '📄 Content & Writing Tools',
            'apps': ['news_summarizer', 'content_writer', 'text_summarizer']
        },
        'development': {
            'title': '💻 Development Tools', 
            'apps': ['code_generator']
        },
        'analysis': {
            'title': '🔍 Analysis Tools',
            'apps': ['legal_analyzer']
        },
        'editing': {
            'title': '✏️ Editing Tools',
            'apps': ['grammar_checker']
        }
    }
    
    for category_info in categories.values():
        st.markdown(f"### {category_info['title']}")
        
        cols = st.columns(len(category_info['apps']))
        for i, app_key in enumerate(category_info['apps']):
            app_info = APPLICATIONS[app_key]
            
            with cols[i]:
                if st.button(
                    f"{app_info['icon']} {app_info['title']}", 
                    key=f"main_{app_key}",
                    use_container_width=True,
                    help=app_info['description']
                ):
                    st.session_state.current_app = app_key
                    st.rerun()
                
                st.caption(app_info['description'])
                if 'model' in app_info:
                    st.caption(f"🎯 Uses: {app_info['model']}")
        
        st.markdown("---")
    
    # Quick stats
    if st.session_state.get('processing_history'):
        st.markdown("## 📊 Usage Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        history = st.session_state.processing_history
        with col1:
            st.metric("Total Requests", len(history))
        with col2:
            successful = sum(1 for h in history if h.get('success', False))
            st.metric("Success Rate", f"{successful/len(history)*100:.1f}%")
        with col3:
            if history:
                avg_time = sum(h.get('response_time', 0) for h in history) / len(history)
                st.metric("Avg Response", f"{avg_time:.2f}s")
        with col4:
            models_used = len(set(h.get('model', '') for h in history))
            st.metric("Models Used", models_used)

# News Summarizer with enhanced features
def render_news_summarizer():
    """Enhanced news summarizer with better UI and error handling."""
    st.title("📰 News Summarizer")
    st.markdown("Get the latest news with AI-powered summaries")
    
    # Configuration panel
    col_config, col_action = st.columns([2, 1])
    
    with col_config:
        categories = ["general", "technology", "business", "entertainment", "health", "science", "sports"]
        selected_category = st.selectbox("News Category", categories, index=1)
        num_articles = st.slider("Number of Articles", 1, 20, 5)
        
        # Country selection (optional enhancement)
        countries = {
            "us": "United States", "gb": "United Kingdom", "ca": "Canada", 
            "au": "Australia", "de": "Germany", "fr": "France"
        }
        selected_country = st.selectbox("Country", list(countries.keys()), 
                                      format_func=lambda x: countries[x], index=0)
    
    with col_action:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        fetch_button = st.button("🔄 Fetch & Summarize News", type="primary", use_container_width=True)
        
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.news_data = None
            st.session_state.summary = None
            st.rerun()
    
    # Main processing
    if fetch_button:
        if not NEWS_API_KEY:
            st.error("⚠️ Please set your NEWS_API_KEY in the .env file!")
            st.info("Get your free API key from [NewsAPI.org](https://newsapi.org/) and add it to your .env file")
        else:
            with st.spinner("Fetching latest news..."):
                try:
                    params = {
                        "category": selected_category,
                        "country": selected_country,
                        "language": "en",
                        "apiKey": NEWS_API_KEY,
                        "pageSize": num_articles
                    }
                    
                    news_response = requests.get(NEWS_API_URL, params=params, timeout=10)
                    
                    if news_response.status_code == 200:
                        news_data = news_response.json()
                        if "articles" in news_data and news_data["articles"]:
                            articles = news_data["articles"][:num_articles]
                            st.session_state.news_data = articles
                            
                            # Prepare text for summarization
                            news_text = "\n".join([
                                f"- {article['title']} ({article['source']['name']}): {article.get('description', 'No description')}"
                                for article in articles
                            ])
                            
                            # Summarize with Ollama
                            model_key = st.session_state.get('selected_model', 'mistral:latest')
                            prompt = f"""Analyze and summarize these {selected_category} news articles from {countries[selected_country]}:

{news_text}

Please provide:
1. A brief overview of the main topics
2. Key trends and patterns
3. Most important stories
4. Any notable developments

Keep the summary informative but concise."""

                            success, summary, response_time = make_ollama_request(prompt, model_key)
                            
                            if success:
                                st.session_state.news_summary = summary
                                st.success(f"✅ News fetched and summarized in {response_time:.2f}s!")
                            else:
                                st.error(f"Failed to generate summary: {summary}")
                        else:
                            st.warning("No news articles found for this category and country.")
                    else:
                        st.error(f"Failed to fetch news: HTTP {news_response.status_code}")
                        
                except requests.exceptions.Timeout:
                    st.error("⏰ Request timed out. Please try again.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Display results
    if hasattr(st.session_state, 'news_summary'):
        st.markdown("## 🤖 AI Summary")
        st.markdown(f'<div class="summary-box">{st.session_state.news_summary}</div>', 
                   unsafe_allow_html=True)
    
    if st.session_state.news_data:
        st.markdown("## 📑 Articles")
        for i, article in enumerate(st.session_state.news_data, 1):
            title = article.get('title', 'No title')
            source = article.get('source', {}).get('name', 'Unknown')
            description = article.get('description', 'No description')
            url = article.get('url', '#')
            published_at = article.get('publishedAt', '')
            
            # Format date
            if published_at:
                try:
                    pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    formatted_date = pub_date.strftime('%Y-%m-%d %H:%M')
                except:
                    formatted_date = published_at
            else:
                formatted_date = 'Unknown date'
            
            st.markdown(f"""
            <div class="news-card">
                <h4>{i}. {title}</h4>
                <p><em>{source} • {formatted_date}</em></p>
                <p>{description}</p>
                <a href="{url}" target="_blank" style="color: #1f77b4; text-decoration: none;">🔗 Read full article</a>
            </div>
            """, unsafe_allow_html=True)

# Code Generator with enhanced features
def render_code_generator():
    """Enhanced code generator with better interface."""
    st.title("💻 Code Generator & Debugger")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 💬 Input")
        
        # Enhanced form
        with st.form("code_form"):
            mode = st.radio("Mode", ["generate", "debug", "optimize", "explain"], horizontal=True)
            
            if mode == "generate":
                prompt = st.text_area("Describe what you want to generate:", height=150,
                                    placeholder="Example: Create a Python function to calculate Fibonacci numbers with memoization")
                language = st.selectbox("Language", ["Python", "JavaScript", "Java", "C++", "Go", "Rust"])
                include_tests = st.checkbox("Include unit tests")
                
            elif mode == "debug":
                prompt = st.text_area("Paste your code to debug:", height=200,
                                    placeholder="Paste your buggy code here...")
                error_msg = st.text_input("Error message (optional):", 
                                         placeholder="What error are you getting?")
                
            elif mode == "optimize":
                prompt = st.text_area("Paste your code to optimize:", height=200,
                                    placeholder="Paste code that needs performance optimization...")
                focus_area = st.selectbox("Optimization focus", 
                                        ["Performance", "Memory usage", "Readability", "Best practices"])
                
            else:  # explain
                prompt = st.text_area("Paste code to explain:", height=200,
                                    placeholder="Paste code you want explained...")
                detail_level = st.selectbox("Detail level", ["Beginner", "Intermediate", "Advanced"])
            
            submitted = st.form_submit_button("🚀 Process", type="primary")
        
        if submitted and prompt.strip():
            model_key = st.session_state.get('selected_model', 'codellama:13b')
            
            # Construct prompts based on mode
            if mode == "generate":
                full_prompt = f"""Generate clean, well-documented {language} code for: {prompt}

Requirements:
- Use best practices and proper naming conventions
- Include comprehensive comments
- Handle edge cases appropriately
{f"- Include unit tests" if include_tests else ""}

Provide complete, working code that can be run immediately."""

            elif mode == "debug":
                full_prompt = f"""Debug and fix this code:

```
{prompt}
```

{f"Error message: {error_msg}" if error_msg else ""}

Please:
1. Identify the bug(s)
2. Explain what's wrong
3. Provide the corrected code
4. Explain the fix"""

            elif mode == "optimize":
                full_prompt = f"""Optimize this code for {focus_area.lower()}:

```
{prompt}
```

Please:
1. Analyze current performance/issues
2. Suggest specific optimizations
3. Provide optimized version
4. Explain the improvements"""

            else:  # explain
                full_prompt = f"""Explain this code at a {detail_level.lower()} level:

```
{prompt}
```

Please provide:
1. Overall purpose and functionality
2. Step-by-step breakdown
3. Key concepts and techniques used
4. Any notable patterns or practices"""
            
            with st.spinner(f"🧠 AI is {mode}ing your code..."):
                success, result, response_time = make_ollama_request(full_prompt, model_key)
                
                if success:
                    st.session_state.canvas_content = result
                    st.success(f"✅ Code {mode}d successfully in {response_time:.2f}s!")
                else:
                    st.error(result)
    
    with col2:
        st.markdown("### 📝 Output")
        
        # Action buttons
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn1:
            if st.button("📋 Copy", help="Copy to clipboard"):
                st.code(st.session_state.canvas_content)
                st.toast("Code ready to copy!")
        with col_btn2:
            if st.button("💾 Save", help="Save to file"):
                if st.session_state.canvas_content:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        "Download",
                        st.session_state.canvas_content,
                        f"code_{timestamp}.txt",
                        mime="text/plain"
                    )
        with col_btn3:
            if st.button("🗑️ Clear"):
                st.session_state.canvas_content = ""
                st.rerun()
        
        # Enhanced text area with syntax highlighting hint
        canvas_content = st.text_area(
            "Generated Code",
            value=st.session_state.canvas_content,
            height=450,
            key="code_canvas",
            help="Edit the generated code here"
        )
        
        if canvas_content != st.session_state.canvas_content:
            st.session_state.canvas_content = canvas_content
        
        # Code stats
        if st.session_state.canvas_content:
            lines = len(st.session_state.canvas_content.split('\n'))
            chars = len(st.session_state.canvas_content)
            words = len(st.session_state.canvas_content.split())
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Lines", lines)
            with col_stat2:
                st.metric("Words", words)
            with col_stat3:
                st.metric("Characters", chars)

# Enhanced Legal Document Analyzer
def render_legal_analyzer():
    """Enhanced legal document analyzer with better file handling."""
    st.title("📄 Legal Document Analyzer")
    st.markdown("Upload and analyze legal documents for key insights, risks, and obligations")
    
    # File upload section
    upload_col, text_col = st.columns([1, 1])
    
    with upload_col:
        st.markdown("#### 📁 File Upload")
        
        # Show available file types
        supported_types = ["txt"]
        if DOCX_AVAILABLE:
            supported_types.append("docx")
        if PYPDF2_AVAILABLE:
            supported_types.append("pdf")
        
        st.info(f"📋 Supported formats: {', '.join(supported_types).upper()}")
        
        uploaded_file = st.file_uploader("Choose a document", type=supported_types)
        
        file_text = ""
        if uploaded_file is not None:
            filetype = uploaded_file.name.split(".")[-1].lower()
            
            try:
                with st.spinner("📖 Extracting text from document..."):
                    if filetype == "txt":
                        file_text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
                        
                    elif filetype == "pdf" and PYPDF2_AVAILABLE:
                        try:
                            pdf_reader = PyPDF2.PdfReader(uploaded_file)
                            pages = []
                            for page in pdf_reader.pages:
                                text = page.extract_text()
                                if text:
                                    pages.append(text)
                            file_text = "\n".join(pages)
                        except Exception as e:
                            st.error(f"PDF extraction failed: {str(e)}")
                            
                    elif filetype == "docx" and DOCX_AVAILABLE:
                        try:
                            doc = docx.Document(uploaded_file)
                            file_text = "\n".join([p.text for p in doc.paragraphs])
                        except Exception as e:
                            st.error(f"DOCX extraction failed: {str(e)}")
                    else:
                        st.error("File type not supported or required library not installed")
                
                if file_text:
                    # Show document stats
                    word_count = len(file_text.split())
                    char_count = len(file_text)
                    
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("Words", word_count)
                    with col_stat2:
                        st.metric("Characters", char_count)
                    
                    st.success("✅ Document processed successfully!")
                    
            except Exception as e:
                st.error(f"Failed to extract text: {str(e)}")
    
    with text_col:
        st.markdown("#### ✏️ Direct Text Input")
        user_text = st.text_area("Or paste legal document text", 
                                value=file_text, height=300,
                                placeholder="Paste your legal document text here...")
    
    # Analysis options
    st.markdown("#### 🔍 Analysis Options")
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    
    with col_opt1:
        analysis_type = st.selectbox("Analysis Type", 
                                   ["Comprehensive", "Risk Assessment", "Key Clauses", "Obligations"])
    with col_opt2:
        detail_level = st.selectbox("Detail Level", ["Summary", "Detailed", "Comprehensive"])
    with col_opt3:
        focus_areas = st.multiselect("Focus Areas", 
                                   ["Liability", "Termination", "Payment Terms", "Confidentiality", 
                                    "Intellectual Property", "Compliance"])
    
    # Analysis button
    if st.button("🔍 Analyze Document", type="primary", use_container_width=True):
        final_text = user_text if user_text.strip() else file_text
        
        if final_text.strip():
            model_key = st.session_state.get('selected_model', 'phi4:latest')
            
            # Construct analysis prompt
            focus_text = f" Pay special attention to: {', '.join(focus_areas)}." if focus_areas else ""
            
            prompt = f"""As a legal expert, analyze this legal document with a {detail_level.lower()} level of analysis focusing on {analysis_type.lower()}:

{final_text}

{focus_text}

Please provide:
1. **Document Summary**: Brief overview of the document type and purpose
2. **Key Legal Points**: Most important clauses and provisions
3. **Risk Assessment**: Potential legal risks and liabilities
4. **Obligations**: Key obligations for each party
5. **Recommendations**: Suggestions for review or negotiation

Structure your analysis clearly and highlight critical issues."""
            
            with st.spinner("🧠 AI analyzing document..."):
                success, analysis, response_time = make_ollama_request(prompt, model_key)
                
                if success:
                    st.markdown("## 📋 Analysis Results")
                    st.markdown(f'<div class="summary-box">{analysis}</div>', 
                               unsafe_allow_html=True)
                    
                    # Download option
                    st.download_button(
                        "💾 Download Analysis",
                        analysis,
                        f"legal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                    
                    st.success(f"✅ Analysis completed in {response_time:.2f}s")
                else:
                    st.error(f"Analysis failed: {analysis}")
        else:
            st.warning("⚠️ Please upload a document or paste text to analyze")

# Enhanced Content Writer
def render_content_writer():
    """Enhanced content writer with multiple options."""
    st.title("✍️ Content Writer")
    st.markdown("Generate engaging content with AI assistance")
    
    # Content type selection
    content_types = {
        "Blog Post": "blog_post",
        "Article": "article", 
        "Product Description": "product",
        "Social Media Post": "social",
        "Email": "email",
        "Press Release": "press"
    }
    
    col_type, col_style = st.columns(2)
    
    with col_type:
        content_type = st.selectbox("Content Type", list(content_types.keys()))
    
    with col_style:
        writing_styles = ["informative", "casual", "professional", "persuasive", "humorous", "technical", "creative"]
        style = st.selectbox("Writing Style", writing_styles)
    
    # Main form
    with st.form("writing_form"):
        topic = st.text_input("Topic/Subject", placeholder="Enter the main topic or subject")
        
        col_details1, col_details2 = st.columns(2)
        with col_details1:
            target_audience = st.text_input("Target Audience", placeholder="Who is this for?")
            word_count = st.selectbox("Approximate Length", 
                                    ["Short (200-400 words)", "Medium (400-800 words)", "Long (800-1500 words)"])
        
        with col_details2:
            tone = st.selectbox("Tone", ["Neutral", "Positive", "Authoritative", "Friendly", "Urgent"])
            include_cta = st.checkbox("Include Call-to-Action")
        
        additional_notes = st.text_area("Additional Requirements", 
                                       placeholder="Any specific points to cover, keywords to include, etc.")
        
        submitted = st.form_submit_button("✨ Generate Content", type="primary")
    
    if submitted and topic and style:
        model_key = st.session_state.get('selected_model', 'llama3.1:8b')
        
        # Construct comprehensive prompt
        length_guide = word_count.split("(")[1].split(")")[0]
        
        prompt = f"""Create a {content_type.lower()} about "{topic}" with the following specifications:

Content Type: {content_type}
Writing Style: {style}
Tone: {tone}
Target Audience: {target_audience or 'General audience'}
Length: {length_guide}

{f"Additional Requirements: {additional_notes}" if additional_notes else ""}

Please create engaging, well-structured content that:
1. Has a compelling introduction
2. Provides valuable information
3. Uses appropriate formatting (headings, paragraphs)
4. Maintains the requested tone and style
{f"5. Includes a compelling call-to-action" if include_cta else ""}

Make it ready to publish!"""
        
        with st.spinner("✍️ AI is crafting your content..."):
            success, content, response_time = make_ollama_request(prompt, model_key, temperature=0.8)
            
            if success:
                st.markdown("## 📝 Generated Content")
                st.markdown(f'<div class="summary-box">{content}</div>', unsafe_allow_html=True)
                
                # Content actions
                col_action1, col_action2, col_action3 = st.columns(3)
                
                with col_action1:
                    word_count_actual = len(content.split())
                    st.metric("Word Count", word_count_actual)
                
                with col_action2:
                    st.download_button(
                        "💾 Download",
                        content,
                        f"{content_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                with col_action3:
                    reading_time = max(1, word_count_actual // 200)
                    st.metric("Reading Time", f"{reading_time} min")
                
                st.success(f"✅ Content generated in {response_time:.2f}s!")
            else:
                st.error(f"Content generation failed: {content}")

# Enhanced Grammar Checker
def render_grammar_checker():
    """Enhanced grammar checker with detailed feedback."""
    st.title("📝 Grammar & Spell Checker")
    st.markdown("Proofread and perfect your text with AI assistance")
    
    # Input options
    input_method = st.radio("Input Method", ["Type/Paste Text", "Upload File"], horizontal=True)
    
    user_text = ""
    
    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
        if uploaded_file:
            user_text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
    
    with st.form("grammar_form"):
        text_input = st.text_area("Text to proofread", value=user_text, height=250,
                                placeholder="Paste or type the text you want to proofread...")
        
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            check_level = st.selectbox("Check Level", 
                                     ["Basic (Grammar & Spelling)", "Standard (+ Style)", "Advanced (+ Clarity)"])
        with col_opt2:
            writing_context = st.selectbox("Writing Context", 
                                         ["General", "Academic", "Business", "Creative", "Technical"])
        
        submitted = st.form_submit_button("🔍 Check Text", type="primary")
    
    if submitted and text_input.strip():
        model_key = st.session_state.get('selected_model', 'mistral:latest')
        
        # Construct detailed prompt based on check level
        if "Basic" in check_level:
            focus = "grammar and spelling errors"
        elif "Standard" in check_level:
            focus = "grammar, spelling, and style issues"
        else:
            focus = "grammar, spelling, style, and clarity improvements"
        
        prompt = f"""As an expert editor, carefully proofread this {writing_context.lower()} text for {focus}:

"{text_input}"

Please provide:
1. **Corrected Version**: The text with all corrections applied
2. **Issues Found**: List specific problems identified
3. **Improvements Made**: Explanation of key changes
4. **Writing Quality**: Overall assessment and suggestions

Focus on maintaining the original meaning while improving clarity and correctness."""
        
        with st.spinner("🔍 AI is proofreading your text..."):
            success, result, response_time = make_ollama_request(prompt, model_key)
            
            if success:
                st.markdown("## 📋 Proofreading Results")
                
                # Split view for comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📄 Original Text")
                    st.text_area("", value=text_input, height=300, disabled=True, key="original")
                
                with col2:
                    st.markdown("### ✅ AI Analysis & Corrections")
                    st.markdown(f'<div class="summary-box">{result}</div>', unsafe_allow_html=True)
                
                # Text statistics
                st.markdown("### 📊 Text Statistics")
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                original_stats = {
                    'words': len(text_input.split()),
                    'characters': len(text_input),
                    'sentences': len([s for s in text_input.split('.') if s.strip()]),
                    'paragraphs': len([p for p in text_input.split('\n\n') if p.strip()])
                }
                
                with col_stat1:
                    st.metric("Words", original_stats['words'])
                with col_stat2:
                    st.metric("Characters", original_stats['characters'])
                with col_stat3:
                    st.metric("Sentences", original_stats['sentences'])
                with col_stat4:
                    st.metric("Paragraphs", original_stats['paragraphs'])
                
                # Download option
                st.download_button(
                    "💾 Download Analysis",
                    result,
                    f"grammar_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
                st.success(f"✅ Proofreading completed in {response_time:.2f}s!")
            else:
                st.error(f"Proofreading failed: {result}")

# Enhanced Text Summarizer
def render_text_summarizer():
    """Enhanced text summarizer with multiple options."""
    st.title("📋 Text Summarizer")
    st.markdown("Create intelligent summaries of long texts")
    
    # Input options
    col_input, col_options = st.columns([2, 1])
    
    with col_input:
        input_method = st.radio("Input Method", ["Paste Text", "Upload File"], horizontal=True)
        
        user_text = ""
        if input_method == "Upload File":
            uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
            if uploaded_file:
                user_text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
                
                # Show file stats
                if user_text:
                    words = len(user_text.split())
                    st.info(f"📊 File loaded: {words} words")
    
    with col_options:
        st.markdown("#### ⚙️ Summary Options")
        summary_length = st.selectbox("Summary Length", 
                                    ["Brief (1-2 paragraphs)", "Standard (3-4 paragraphs)", "Detailed (5+ paragraphs)"])
        summary_style = st.selectbox("Summary Style", 
                                   ["Bullet Points", "Paragraph Form", "Executive Summary"])
        focus_areas = st.multiselect("Focus On", 
                                   ["Key Points", "Statistics", "Conclusions", "Recommendations", "Timeline"])
    
    # Main form
    with st.form("summarize_form"):
        text_input = st.text_area("Enter text to summarize", value=user_text, height=300,
                                placeholder="Paste your long text here to get an intelligent summary...")
        
        submitted = st.form_submit_button("📝 Generate Summary", type="primary")
    
    if submitted and text_input.strip():
        model_key = st.session_state.get('selected_model', 'mistral:latest')
        
        # Calculate input stats
        input_words = len(text_input.split())
        input_chars = len(text_input)
        
        if input_words < 50:
            st.warning("⚠️ Text seems quite short. Summaries work best with longer texts (100+ words).")
            return
        
        # Construct detailed prompt
        length_instruction = summary_length.split("(")[1].split(")")[0]
        focus_text = f" Focus especially on: {', '.join(focus_areas)}." if focus_areas else ""
        
        prompt = f"""Create a {summary_style.lower()} summary of the following text:

{text_input}

Summary requirements:
- Length: {length_instruction}
- Style: {summary_style}
- Capture the main ideas and key information
{focus_text}

Make the summary comprehensive yet concise, ensuring no critical information is lost."""
        
        with st.spinner("📝 AI is analyzing and summarizing..."):
            success, summary, response_time = make_ollama_request(prompt, model_key)
            
            if success:
                st.markdown("## 📄 Summary Results")
                
                # Summary display
                st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                
                # Comparison stats
                st.markdown("### 📊 Summary Statistics")
                summary_words = len(summary.split())
                summary_chars = len(summary)
                compression_ratio = (1 - summary_words / input_words) * 100
                
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                with col_stat1:
                    st.metric("Original Words", input_words)
                with col_stat2:
                    st.metric("Summary Words", summary_words)
                with col_stat3:
                    st.metric("Compression", f"{compression_ratio:.1f}%")
                with col_stat4:
                    reading_time = max(1, summary_words // 200)
                    st.metric("Reading Time", f"{reading_time} min")
                
                # Download option
                st.download_button(
                    "💾 Download Summary",
                    summary,
                    f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
                st.success(f"✅ Summary generated in {response_time:.2f}s!")
            else:
                st.error(f"Summarization failed: {summary}")

# Main application router with enhanced error handling
def main():
    """Main application with enhanced routing and error handling."""
    
    # Initialize session state
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Route to appropriate application
    current_app = st.session_state.current_app
    
    try:
        if current_app == 'dashboard':
            render_dashboard()
        elif current_app == 'news_summarizer':
            render_news_summarizer()
        elif current_app == 'code_generator':
            render_code_generator()
        elif current_app == 'legal_analyzer':
            render_legal_analyzer()
        elif current_app == 'content_writer':
            render_content_writer()
        elif current_app == 'grammar_checker':
            render_grammar_checker()
        elif current_app == 'text_summarizer':
            render_text_summarizer()
        else:
            st.error(f"Unknown application: {current_app}")
            st.session_state.current_app = 'dashboard'
            st.rerun()
            
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Returning to dashboard...")
        st.session_state.current_app = 'dashboard'
        if st.button("🔄 Reload Dashboard"):
            st.rerun()
    
    # Enhanced footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; padding: 2rem;">
        <p>🤖 <strong>AI Word Processing Suite v2.0</strong> | 
        Powered by <a href="https://ollama.com/" target="_blank" style="color: #667eea;">Ollama</a> & 
        <a href="https://streamlit.io/" target="_blank" style="color: #667eea;">Streamlit</a></p>
        <p>💡 <em>Professional AI-powered text processing with privacy-first local processing</em></p>
        <p>📊 Session Stats: {len(st.session_state.get('processing_history', []))} requests processed</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()