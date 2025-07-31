# AI News Summarizer 
import streamlit as st
import requests
import json
from datetime import datetime
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
NEWS_API_URL = "https://newsapi.org/v2/top-headlines"
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OLLAMA_URL = "http://localhost:11434/api/generate"

# Available models based on your ollama list
AVAILABLE_MODELS = {
    "mistral:latest": "Mistral Latest - Fast and efficient for summarization",
    "llama3.1:8b": "Llama 3.1 8B - Excellent general purpose model",
    "deepseek-r1:latest": "DeepSeek R1 Latest - Advanced reasoning capabilities",
    "deepseek-r1:1.5b": "DeepSeek R1 1.5B - Lightweight but capable",
    "phi4:latest": "Phi4 Latest - Microsoft's efficient model",
    "gemma3:12b": "Gemma3 12B - Google's powerful text model",
    "gemma3n:e4b": "Gemma3N E4B - Specialized Gemma variant"
}

# News categories
NEWS_CATEGORIES = [
    "general", "technology", "business", "entertainment", 
    "health", "science", "sports"
]

# Page configuration
st.set_page_config(
    page_title="AI News Summarizer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.news-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #1f77b4;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.news-title {
    font-size: 1.2rem;
    font-weight: bold;
    color: #1f77b4;
    margin-bottom: 0.5rem;
}
.news-source {
    color: #666;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
}
.news-description {
    color: #333;
    line-height: 1.6;
    margin-bottom: 1rem;
}
.news-content {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 5px;
    border-left: 3px solid #28a745;
    margin: 0.5rem 0;
    max-height: 300px;
    overflow-y: auto;
}
.summary-box {
    background: #f0f8ff;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #4CAF50;
    margin: 1rem 0;
}
.article-actions {
    display: flex;
    gap: 10px;
    margin-top: 1rem;
}
.action-button {
    padding: 0.5rem 1rem;
    border-radius: 5px;
    text-decoration: none;
    font-size: 0.9rem;
    border: none;
    cursor: pointer;
    transition: all 0.3s;
}
.read-more-btn {
    background: #007bff;
    color: white;
}
.external-link-btn {
    background: #28a745;
    color: white;
}
.action-button:hover {
    opacity: 0.8;
    transform: translateY(-2px);
}
.stAlert > div {
    background-color: #d4edda;
    border-color: #c3e6cb;
    color: #155724;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'news_data' not in st.session_state:
    st.session_state.news_data = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = None
if 'article_contents' not in st.session_state:
    st.session_state.article_contents = {}

def fetch_article_content(url):
    """Fetch article content from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            # Simple content extraction (you might want to use BeautifulSoup for better parsing)
            content = response.text
            # Extract a reasonable preview (first 1000 characters after some basic cleaning)
            content = content.replace('<script', '<!-- <script').replace('</script>', '</script> -->')
            if len(content) > 1000:
                content = content[:1000] + "..."
            return content
        else:
            return "Unable to fetch article content."
    except Exception as e:
        return f"Error fetching content: {str(e)}"

# Header
st.title("📰 AI News Summarizer")
st.markdown("Get the latest news with AI-powered summaries using Ollama")

# Sidebar configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Model selection
    selected_model = st.selectbox(
        "Choose AI Model",
        options=list(AVAILABLE_MODELS.keys()),
        format_func=lambda x: AVAILABLE_MODELS[x],
        index=0
    )
    
    # Category selection
    selected_category = st.selectbox(
        "News Category",
        options=NEWS_CATEGORIES,
        index=1  # Default to "technology"
    )
    
    # Number of articles
    num_articles = st.slider(
        "Number of Articles",
        min_value=1,
        max_value=100,
        value=10
    )
    
    st.markdown("---")
    
    # API status check
    if st.button("🔍 Test Ollama Connection"):
        try:
            test_response = requests.post(
                OLLAMA_URL,
                json={"model": selected_model, "prompt": "Hello", "stream": False},
                timeout=5
            )
            if test_response.status_code == 200:
                st.success("✅ Ollama is running!")
            else:
                st.error("❌ Ollama connection failed")
        except Exception as e:
            st.error(f"❌ Cannot connect to Ollama: {str(e)}")

# Main content area
col1, col2 = st.columns([2, 1])

with col2:
    if st.button("🔄 Fetch & Summarize News", type="primary", use_container_width=True):
        if not NEWS_API_KEY:
            st.error("⚠️ Please set your NEWS_API_KEY in the .env file!")
            st.info("Create a .env file with: NEWS_API_KEY=your_actual_api_key")
        else:
            with st.spinner("Fetching latest news..."):
                try:
                    # Fetch news
                    params = {
                        "category": selected_category,
                        "language": "en",
                        "apiKey": NEWS_API_KEY,
                        "pageSize": num_articles
                    }
                    
                    news_response = requests.get(NEWS_API_URL, params=params)
                    
                    if news_response.status_code != 200:
                        st.error(f"Failed to fetch news: {news_response.status_code}")
                    else:
                        news_data = news_response.json()
                        
                        if "articles" not in news_data or not news_data["articles"]:
                            st.warning("No news articles found for this category.")
                        else:
                            articles = news_data["articles"][:num_articles]
                            st.session_state.news_data = articles
                            
                            # Prepare text for summarization
                            news_text = "\n".join([
                                f"- {article['title']} ({article['source']['name']}): {article.get('description', 'No description')}"
                                for article in articles
                            ])
                            
                            # Summarize with Ollama
                            with st.spinner("Generating AI summary..."):
                                ollama_response = requests.post(
                                    OLLAMA_URL,
                                    json={
                                        "model": selected_model, 
                                        "prompt": f"Summarize these {selected_category} news articles in a clear, concise way. Highlight key trends and important information:\n\n{news_text}",
                                        "stream": False
                                    },
                                    timeout=30
                                )
                                
                                if ollama_response.status_code == 200:
                                    try:
                                        json_response = json.loads(ollama_response.text.strip())
                                        summary = json_response.get("response", "No summary available.")
                                        st.session_state.summary = summary
                                        st.session_state.last_fetch_time = datetime.now()
                                        st.success("✅ News fetched and summarized successfully!")
                                    except json.JSONDecodeError:
                                        st.error("Failed to parse Ollama response")
                                else:
                                    st.error(f"Ollama request failed: {ollama_response.status_code}")
                                    
                except requests.exceptions.RequestException as e:
                    st.error(f"Network error: {str(e)}")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")

with col1:
    # Display last update time
    if st.session_state.last_fetch_time:
        st.caption(f"Last updated: {st.session_state.last_fetch_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Display results
if st.session_state.summary:
    st.markdown("## 🤖 AI Summary")
    st.markdown(f"""
    <div class="summary-box">
        <strong>Model Used:</strong> {AVAILABLE_MODELS[selected_model]}<br><br>
        {st.session_state.summary}
    </div>
    """, unsafe_allow_html=True)

if st.session_state.news_data:
    st.markdown("## 📑 Latest Articles")
    
    for i, article in enumerate(st.session_state.news_data, 1):
        title = article.get('title', 'No title')
        source = article.get('source', {}).get('name', 'Unknown source')
        description = article.get('description', 'No description available')
        content = article.get('content', 'No content available')
        url = article.get('url', '#')
        url_to_image = article.get('urlToImage', '')
        published_at = article.get('publishedAt', '')
        
        # Format published date
        if published_at:
            try:
                pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                formatted_date = pub_date.strftime('%Y-%m-%d %H:%M')
            except:
                formatted_date = published_at
        else:
            formatted_date = 'Unknown date'
        
        # Create article container
        with st.container():
            # Article header
            st.markdown(f"""
            <div class="news-card">
                <div class="news-title">{i}. {title}</div>
                <div class="news-source">{source} • {formatted_date}</div>
                <div class="news-description">{description}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Article image if available
            if url_to_image:
                try:
                    st.image(url_to_image, caption=f"Image from {source}", use_container_width=True)
                except:
                    pass  # Skip if image fails to load
            
            # Article actions
            col_actions1, col_actions2, col_actions3 = st.columns([1, 1, 2])
            
            with col_actions1:
                # Preview article content
                if st.button(f"📖 Preview", key=f"preview_{i}"):
                    article_key = f"article_{i}"
                    if article_key not in st.session_state.article_contents:
                        with st.spinner("Fetching article content..."):
                            st.session_state.article_contents[article_key] = fetch_article_content(url)
            
            with col_actions2:
                # External link button with JavaScript to open in new tab
                st.markdown(f"""
                <a href="{url}" target="_blank" rel="noopener noreferrer" 
                   style="display: inline-block; padding: 0.375rem 0.75rem; 
                          background: #28a745; color: white; text-decoration: none; 
                          border-radius: 0.25rem; font-size: 0.875rem;">
                    🔗 Open Article
                </a>
                """, unsafe_allow_html=True)
            
            # Display article content preview if fetched
            article_key = f"article_{i}"
            if article_key in st.session_state.article_contents:
                with st.expander(f"📄 Article Preview - {title[:50]}...", expanded=True):
                    # Show original content from NewsAPI
                    if content and content.strip() and content != 'No content available':
                        st.markdown("**Article Content (from NewsAPI):**")
                        st.markdown(f"""
                        <div class="news-content">
                            {content}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show fetched content
                    fetched_content = st.session_state.article_contents[article_key]
                    if fetched_content and "Error fetching content" not in fetched_content:
                        st.markdown("**Fetched Content Preview:**")
                        st.markdown(f"""
                        <div class="news-content">
                            {fetched_content[:500]}...
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("Unable to fetch additional content from the source.")
                    
                    # Clear preview button
                    if st.button(f"❌ Close Preview", key=f"close_{i}"):
                        del st.session_state.article_contents[article_key]
                        st.rerun()
            
            st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    Powered by NewsAPI & Ollama | Built with Streamlit<br>
    💡 Tip: Make sure Ollama is running locally with your selected model
</div>
""", unsafe_allow_html=True)