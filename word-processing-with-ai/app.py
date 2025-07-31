# AI Word Processing Suite - Main Dashboard
import streamlit as st
import requests
import json
from datetime import datetime
import time
import os
import io
from io import StringIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import optional dependencies
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

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
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

# Custom CSS for the dashboard
st.markdown("""
<style>
.app-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    margin: 1rem 0;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.app-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.2);
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
    padding: 2rem 0;
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    border-radius: 15px;
    margin-bottom: 2rem;
}
.status-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 8px;
}
.status-online {
    background-color: #4CAF50;
}
.status-offline {
    background-color: #f44336;
}
.news-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #1f77b4;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.summary-box {
    background: #f0f8ff;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #4CAF50;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_app' not in st.session_state:
    st.session_state.current_app = 'dashboard'
if 'canvas_content' not in st.session_state:
    st.session_state.canvas_content = ""
if 'news_data' not in st.session_state:
    st.session_state.news_data = None
if 'article_contents' not in st.session_state:
    st.session_state.article_contents = {}

# Check Ollama connection
def check_ollama_status():
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": "mistral:latest", "prompt": "Hello", "stream": False},
            timeout=3
        )
        return response.status_code == 200
    except:
        return False

# Application definitions
APPLICATIONS = {
    'dashboard': {
        'title': 'Dashboard',
        'description': 'Main dashboard and application selector',
        'icon': '🏠'
    },
    'news_summarizer': {
        'title': 'News Summarizer',
        'description': 'Fetch and summarize latest news articles with AI',
        'icon': '📰',
        'model': 'mistral:latest'
    },
    'code_generator': {
        'title': 'Code Generator & Debugger',
        'description': 'Generate code and debug existing code with AI assistance',
        'icon': '💻',
        'model': 'codellama:13b'
    },
    'legal_analyzer': {
        'title': 'Legal Document Analyzer',
        'description': 'Analyze legal documents for key insights, risks, and obligations',
        'icon': '📄',
        'model': 'phi4:latest'
    },
    'content_writer': {
        'title': 'Content Writer',
        'description': 'Generate blog posts and articles with various writing styles',
        'icon': '✍️',
        'model': 'llama3.1:8b'
    },
    'grammar_checker': {
        'title': 'Grammar & Spell Checker',
        'description': 'Proofread and correct grammar, spelling, and sentence structure',
        'icon': '📝',
        'model': 'mistral:latest'
    },
    'text_summarizer': {
        'title': 'Text Summarizer',
        'description': 'Create concise summaries of long texts',
        'icon': '📋',
        'model': 'mistral:latest'
    }
}

# Sidebar navigation
with st.sidebar:
    st.markdown("### 🚀 AI Word Processing Suite")
    
    # Ollama status
    ollama_online = check_ollama_status()
    status_class = "status-online" if ollama_online else "status-offline"
    status_text = "Online" if ollama_online else "Offline"
    
    st.markdown(f"""
    <div style="margin: 1rem 0;">
        <strong>Ollama Status:</strong><br>
        <span class="status-indicator {status_class}"></span>{status_text}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 📱 Applications")
    for app_key, app_info in APPLICATIONS.items():
        if st.button(f"{app_info['icon']} {app_info['title']}", 
                    use_container_width=True,
                    key=f"nav_{app_key}"):
            st.session_state.current_app = app_key
            st.rerun()
    
    st.markdown("---")
    
    # Model selection for current app
    if st.session_state.current_app != 'dashboard':
        current_app_info = APPLICATIONS[st.session_state.current_app]
        if 'model' in current_app_info:
            st.markdown("### ⚙️ Model Selection")
            default_model = current_app_info['model']
            selected_model = st.selectbox(
                "Choose AI Model",
                options=list(AVAILABLE_MODELS.keys()),
                format_func=lambda x: AVAILABLE_MODELS[x],
                index=list(AVAILABLE_MODELS.keys()).index(default_model) if default_model in AVAILABLE_MODELS else 0,
                key="model_selector"
            )
            st.session_state.selected_model = selected_model

# Main content area
def render_dashboard():
    st.markdown("""
    <div class="dashboard-header">
        <h1>🤖 AI Word Processing Suite</h1>
        <p>Choose an application below to get started with AI-powered text processing</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a grid of applications
    col1, col2 = st.columns(2)
    
    apps_list = list(APPLICATIONS.items())[1:]  # Skip dashboard
    
    for i, (app_key, app_info) in enumerate(apps_list):
        col = col1 if i % 2 == 0 else col2
        
        with col:
            if st.button(f"{app_info['icon']} {app_info['title']}", 
                        key=f"main_{app_key}",
                        use_container_width=True):
                st.session_state.current_app = app_key
                st.rerun()
            
            st.markdown(f"**{app_info['description']}**")
            if 'model' in app_info:
                st.caption(f"Recommended model: {app_info['model']}")
            st.markdown("---")

def render_news_summarizer():
    st.title("📰 News Summarizer")
    st.markdown("Get the latest news with AI-powered summaries")
    
    # Configuration
    col1, col2 = st.columns([2, 1])
    
    with col2:
        categories = ["general", "technology", "business", "entertainment", "health", "science", "sports"]
        selected_category = st.selectbox("News Category", categories, index=1)
        num_articles = st.slider("Number of Articles", 1, 10, 5)
        
        if st.button("🔄 Fetch & Summarize News", type="primary", use_container_width=True):
            if not NEWS_API_KEY:
                st.error("⚠️ Please set your NEWS_API_KEY in the .env file!")
            else:
                with st.spinner("Fetching news..."):
                    try:
                        params = {
                            "category": selected_category,
                            "language": "en",
                            "apiKey": NEWS_API_KEY,
                            "pageSize": num_articles
                        }
                        
                        news_response = requests.get(NEWS_API_URL, params=params)
                        
                        if news_response.status_code == 200:
                            news_data = news_response.json()
                            if "articles" in news_data and news_data["articles"]:
                                articles = news_data["articles"][:num_articles]
                                st.session_state.news_data = articles
                                
                                # Summarize with Ollama
                                news_text = "\n".join([
                                    f"- {article['title']} ({article['source']['name']}): {article.get('description', 'No description')}"
                                    for article in articles
                                ])
                                
                                model_key = st.session_state.get('selected_model', 'mistral:latest')
                                ollama_response = requests.post(
                                    OLLAMA_URL,
                                    json={
                                        "model": model_key,
                                        "prompt": f"Summarize these {selected_category} news articles:\n\n{news_text}",
                                        "stream": False
                                    },
                                    timeout=30
                                )
                                
                                if ollama_response.status_code == 200:
                                    json_response = json.loads(ollama_response.text.strip())
                                    st.session_state.news_summary = json_response.get("response", "No summary available.")
                                    st.success("✅ News fetched and summarized!")
                        else:
                            st.error(f"Failed to fetch news: {news_response.status_code}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with col1:
        # Display summary
        if hasattr(st.session_state, 'news_summary'):
            st.markdown("## 🤖 AI Summary")
            st.markdown(f'<div class="summary-box">{st.session_state.news_summary}</div>', unsafe_allow_html=True)
        
        # Display articles
        if st.session_state.news_data:
            st.markdown("## 📑 Articles")
            for i, article in enumerate(st.session_state.news_data, 1):
                title = article.get('title', 'No title')
                source = article.get('source', {}).get('name', 'Unknown')
                description = article.get('description', 'No description')
                url = article.get('url', '#')
                
                st.markdown(f"""
                <div class="news-card">
                    <h4>{i}. {title}</h4>
                    <p><em>{source}</em></p>
                    <p>{description}</p>
                    <a href="{url}" target="_blank">🔗 Read full article</a>
                </div>
                """, unsafe_allow_html=True)

def render_code_generator():
    st.title("💻 Code Generator & Debugger")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 💬 Input")
        
        with st.form("code_form"):
            mode = st.radio("Mode", ["generate", "debug"], horizontal=True)
            prompt = st.text_area("Prompt", height=200)
            submitted = st.form_submit_button("Run")
        
        if submitted and prompt.strip():
            model_key = st.session_state.get('selected_model', 'codellama:13b')
            
            if mode == "generate":
                full_prompt = f"Write clean, well-documented code for: {prompt}"
            else:
                full_prompt = f"Debug and fix this code:\n{prompt}"
            
            with st.spinner("Processing..."):
                try:
                    response = requests.post(
                        OLLAMA_URL,
                        json={
                            "model": model_key,
                            "prompt": full_prompt,
                            "stream": False
                        }
                    )
                    
                    if response.status_code == 200:
                        json_response = json.loads(response.text.strip())
                        code_result = json_response.get("response", "No response")
                        st.session_state.canvas_content = code_result
                        st.success(f"✅ Code {mode}d successfully!")
                    else:
                        st.error("Failed to process request")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        st.markdown("### 📝 Output")
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("📋 Copy"):
                st.toast("Code copied!")
        with col_btn2:
            if st.button("🗑️ Clear"):
                st.session_state.canvas_content = ""
                st.rerun()
        
        canvas_content = st.text_area(
            "Generated Code",
            value=st.session_state.canvas_content,
            height=400,
            key="code_canvas"
        )
        
        if canvas_content != st.session_state.canvas_content:
            st.session_state.canvas_content = canvas_content

def render_legal_analyzer():
    st.title("📄 Legal Document Analyzer")
    st.markdown("Upload and analyze legal documents for key insights")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a document", type=["txt", "pdf", "docx"])
    
    file_text = ""
    if uploaded_file is not None:
        filetype = uploaded_file.name.split(".")[-1].lower()
        try:
            if filetype == "txt":
                file_text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            elif filetype == "pdf" and PYPDF2_AVAILABLE:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                pages = []
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
                file_text = "\n".join(pages)
            elif filetype == "docx" and DOCX_AVAILABLE:
                doc = docx.Document(uploaded_file)
                file_text = "\n".join([p.text for p in doc.paragraphs])
            else:
                st.error("File type not supported or required library not installed")
        except Exception as e:
            st.error(f"Failed to extract text: {str(e)}")
    
    user_text = st.text_area("Or paste legal document text", value=file_text, height=300)
    
    if st.button("🔍 Analyze Document", type="primary"):
        if user_text.strip():
            model_key = st.session_state.get('selected_model', 'phi4:latest')
            prompt = f"Analyze this legal document and extract key insights, risks, and obligations:\n{user_text}"
            
            with st.spinner("Analyzing..."):
                try:
                    response = requests.post(
                        OLLAMA_URL,
                        json={
                            "model": model_key,
                            "prompt": prompt,
                            "stream": False
                        }
                    )
                    
                    if response.status_code == 200:
                        json_response = json.loads(response.text.strip())
                        analysis = json_response.get("response", "No analysis available")
                        
                        st.markdown("## 📋 Analysis Results")
                        st.markdown(f'<div class="summary-box">{analysis}</div>', unsafe_allow_html=True)
                    else:
                        st.error("Failed to analyze document")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please upload a document or paste text")

def render_content_writer():
    st.title("✍️ Content Writer")
    st.markdown("Generate blog posts and articles with AI")
    
    with st.form("writing_form"):
        topic = st.text_input("Topic")
        style = st.selectbox("Writing Style", 
                           ["informative", "casual", "persuasive", "humorous", "technical"])
        submitted = st.form_submit_button("Generate Content", type="primary")
    
    if submitted and topic and style:
        model_key = st.session_state.get('selected_model', 'llama3.1:8b')
        prompt = f"Write a detailed blog post about '{topic}' in a {style} tone."
        
        with st.spinner("Generating content..."):
            try:
                response = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": model_key,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    json_response = json.loads(response.text.strip())
                    content = json_response.get("response", "No content generated")
                    
                    st.markdown("## 📝 Generated Content")
                    st.markdown(f'<div class="summary-box">{content}</div>', unsafe_allow_html=True)
                else:
                    st.error("Failed to generate content")
            except Exception as e:
                st.error(f"Error: {str(e)}")

def render_grammar_checker():
    st.title("📝 Grammar & Spell Checker")
    st.markdown("Proofread and correct your text")
    
    with st.form("grammar_form"):
        user_text = st.text_area("Text to proofread", height=200)
        submitted = st.form_submit_button("Proofread", type="primary")
    
    if submitted and user_text.strip():
        model_key = st.session_state.get('selected_model', 'mistral:latest')
        prompt = f"Correct the grammar, spelling, and sentence structure:\n{user_text}"
        
        with st.spinner("Proofreading..."):
            try:
                response = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": model_key,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    json_response = json.loads(response.text.strip())
                    corrected = json_response.get("response", "No correction available")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Original Text")
                        st.text_area("", value=user_text, height=200, disabled=True)
                    
                    with col2:
                        st.markdown("### Corrected Text")
                        st.text_area("", value=corrected, height=200, key="corrected_output")
                else:
                    st.error("Failed to proofread text")
            except Exception as e:
                st.error(f"Error: {str(e)}")

def render_text_summarizer():
    st.title("📋 Text Summarizer")
    st.markdown("Create concise summaries of long texts")
    
    with st.form("summarize_form"):
        user_text = st.text_area("Enter text to summarize", height=200)
        submitted = st.form_submit_button("Summarize", type="primary")
    
    if submitted and user_text.strip():
        model_key = st.session_state.get('selected_model', 'mistral:latest')
        prompt = f"Summarize this text concisely:\n{user_text}"
        
        with st.spinner("Summarizing..."):
            try:
                response = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": model_key,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    json_response = json.loads(response.text.strip())
                    summary = json_response.get("response", "No summary available")
                    
                    st.markdown("## 📄 Summary")
                    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                else:
                    st.error("Failed to summarize text")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Main app router
def main():
    current_app = st.session_state.current_app
    
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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        🤖 AI Word Processing Suite | Powered by Ollama & Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()