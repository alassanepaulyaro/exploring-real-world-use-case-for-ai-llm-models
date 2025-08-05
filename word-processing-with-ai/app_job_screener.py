# AI Job Application Screener - Fixed Version (No Templates)
import streamlit as st
import requests
import io
import json
import hashlib
import re
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import time

# Enhanced PyMuPDF import with multiple fallback strategies
try:
    import fitz  # PyMuPDF
    # Test if fitz.open is available
    if hasattr(fitz, 'open'):
        PYMUPDF_AVAILABLE = True
        PYMUPDF_VERSION = "standard"
    else:
        # Try alternative import
        import PyMuPDF as fitz
        PYMUPDF_AVAILABLE = True
        PYMUPDF_VERSION = "alternative"
except ImportError:
    try:
        # Try direct PyMuPDF import
        import PyMuPDF as fitz
        PYMUPDF_AVAILABLE = True
        PYMUPDF_VERSION = "direct"
    except ImportError:
        PYMUPDF_AVAILABLE = False
        PYMUPDF_VERSION = "none"

# Import plotly conditionally
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ----------------------------
# CONFIGURATION
# ----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral-nemo:latest"
CACHE_EXPIRY = 3600

# Page configuration
st.set_page_config(
    page_title="AI Job Application Screener",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'resume_history' not in st.session_state:
    st.session_state.resume_history = []
if 'job_history' not in st.session_state:
    st.session_state.job_history = []
if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0

# ----------------------------
# ENHANCED STYLING
# ----------------------------
st.markdown("""
    <style>
    /* Main styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .main-title { 
        font-size: 3rem; 
        font-weight: 800; 
        color: white; 
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .subtitle { 
        font-size: 1.3rem; 
        color: #f0f0f0; 
        margin-bottom: 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Interactive content cards */
    .content-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .content-card:hover {
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    
    /* Content analysis indicators */
    .content-stats {
        display: flex;
        justify-content: space-between;
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .stat-item {
        text-align: center;
        flex: 1;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.25rem;
    }
    
    /* Quality indicators */
    .quality-indicator {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        color: white;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    .quality-excellent { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
    .quality-good { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .quality-fair { background: linear-gradient(135deg, #ffa726 0%, #fb8c00 100%); }
    .quality-poor { background: linear-gradient(135deg, #ef5350 0%, #e53935 100%); }
    
    /* Score visualization */
    .score-container {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .score-title {
        font-size: 1.2rem;
        color: white;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .score-value {
        font-size: 3rem;
        font-weight: 800;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Alternative score visualization */
    .score-bar-container {
        background: #f0f0f0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .score-bar {
        height: 30px;
        border-radius: 15px;
        position: relative;
        overflow: hidden;
        background: #e0e0e0;
    }
    .score-fill {
        height: 100%;
        border-radius: 15px;
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
    .score-excellent { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .score-good { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
    .score-average { background: linear-gradient(135deg, #ffa726 0%, #fb8c00 100%); }
    .score-poor { background: linear-gradient(135deg, #ef5350 0%, #e53935 100%); }
    
    /* Status indicators */
    .status-indicator {
        padding: 0.5rem 1rem;
        border-radius: 25px;
        color: white;
        font-weight: 600;
        text-align: center;
        margin: 0.5rem 0;
    }
    .status-excellent { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .status-good { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
    .status-average { background: linear-gradient(135deg, #ffa726 0%, #fb8c00 100%); }
    .status-poor { background: linear-gradient(135deg, #ef5350 0%, #e53935 100%); }
    
    /* Analysis sections */
    .analysis-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #e0e0e0;
        margin-top: 3rem;
    }
    
    /* Error messages */
    .error-box {
        background: #fee;
        border: 1px solid #fcc;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #c33;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# ENHANCED PDF PROCESSING WITH MULTIPLE STRATEGIES
# ----------------------------
def extract_text_from_pdf(uploaded_file) -> Tuple[str, Dict]:
    """Enhanced PDF text extraction with multiple fallback strategies."""
    if not PYMUPDF_AVAILABLE:
        return "❌ PyMuPDF not available. Install with: pip install PyMuPDF", {"error": True}
    
    try:
        file_content = uploaded_file.read()
        
        # Strategy 1: Standard fitz.open
        if PYMUPDF_VERSION == "standard":
            with fitz.open(stream=file_content, filetype="pdf") as doc:
                text = ""
                metadata = {
                    "pages": len(doc),
                    "file_size": len(file_content),
                    "filename": uploaded_file.name,
                    "method": "standard_fitz"
                }
                
                for page_num, page in enumerate(doc):
                    page_text = page.get_text("text")
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        
        # Strategy 2: Alternative PyMuPDF import
        elif PYMUPDF_VERSION == "alternative":
            doc = fitz.open(stream=file_content, filetype="pdf")
            text = ""
            metadata = {
                "pages": len(doc),
                "file_size": len(file_content),
                "filename": uploaded_file.name,
                "method": "alternative_import"
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text("text")
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            doc.close()
        
        # Strategy 3: Direct PyMuPDF
        else:
            doc = fitz.open(stream=file_content, filetype="pdf")
            text = ""
            metadata = {
                "pages": len(doc),
                "file_size": len(file_content),
                "filename": uploaded_file.name,
                "method": "direct_pymupdf"
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            doc.close()
        
        # Clean and process text
        text = re.sub(r'\n{3,}', '\n\n', text.strip())
        metadata["word_count"] = len(text.split())
        metadata["char_count"] = len(text)
        
        return text if text.strip() else "No text content found in PDF.", metadata
        
    except Exception as e:
        # Try alternative approach if first method fails
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            file_content = uploaded_file.read()
            
            # Alternative approach using PyMuPDF differently
            doc = fitz.Document(stream=file_content, filetype="pdf")
            text = ""
            metadata = {
                "pages": len(doc),
                "file_size": len(file_content),
                "filename": uploaded_file.name,
                "method": "fallback_document"
            }
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            doc.close()
            
            text = re.sub(r'\n{3,}', '\n\n', text.strip())
            metadata["word_count"] = len(text.split())
            metadata["char_count"] = len(text)
            
            return text if text.strip() else "No text content found in PDF.", metadata
            
        except Exception as e2:
            error_msg = f"❌ PDF Processing Error: {str(e)} | Fallback failed: {str(e2)}"
            return error_msg, {"error": True, "original_error": str(e), "fallback_error": str(e2)}

# ----------------------------
# CONTENT ANALYSIS FUNCTIONS
# ----------------------------
def analyze_content_quality(text: str, content_type: str) -> Dict:
    """Analyze content quality and provide recommendations."""
    words = text.split()
    sentences = text.split('.')
    paragraphs = text.split('\n\n')
    
    # Basic metrics
    metrics = {
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "paragraph_count": len([p for p in paragraphs if p.strip()]),
        "avg_words_per_sentence": len(words) / max(len(sentences), 1),
        "reading_time": len(words) / 200
    }
    
    # Quality assessment
    quality_score = 0
    issues = []
    recommendations = []
    
    if content_type == "resume":
        if metrics["word_count"] < 200:
            issues.append("Resume is quite short")
            recommendations.append("Add more detail about your experience and skills")
        elif metrics["word_count"] > 800:
            issues.append("Resume might be too long")
            recommendations.append("Consider condensing to focus on most relevant information")
        else:
            quality_score += 25
            
        key_sections = ["experience", "education", "skills", "project"]
        found_sections = sum(1 for section in key_sections if section in text.lower())
        quality_score += (found_sections / len(key_sections)) * 25
        
        if found_sections < 3:
            issues.append("Missing key resume sections")
            recommendations.append("Include sections for Experience, Education, Skills, and Projects")
            
    elif content_type == "job":
        if metrics["word_count"] < 150:
            issues.append("Job description is quite brief")
            recommendations.append("Add more details about requirements and responsibilities")
        else:
            quality_score += 25
            
        key_elements = ["responsibilities", "requirements", "qualifications", "experience"]
        found_elements = sum(1 for element in key_elements if element in text.lower())
        quality_score += (found_elements / len(key_elements)) * 25
        
        if found_elements < 2:
            issues.append("Missing key job description elements")
            recommendations.append("Include clear responsibilities and requirements")
    
    if metrics["avg_words_per_sentence"] > 20:
        issues.append("Sentences are quite long")
        recommendations.append("Break down complex sentences for better readability")
    else:
        quality_score += 25
        
    if metrics["paragraph_count"] > 1:
        quality_score += 25
    else:
        issues.append("Content lacks proper paragraph structure")
        recommendations.append("Organize content into clear paragraphs")
    
    # Determine quality level
    if quality_score >= 80:
        quality_level = "excellent"
    elif quality_score >= 60:
        quality_level = "good"
    elif quality_score >= 40:
        quality_level = "fair"
    else:
        quality_level = "poor"
    
    return {
        "metrics": metrics,
        "quality_score": quality_score,
        "quality_level": quality_level,
        "issues": issues,
        "recommendations": recommendations
    }

def extract_keywords(text: str, content_type: str) -> List[str]:
    """Extract relevant keywords from content."""
    common_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "a", "an"}
    
    if content_type == "resume":
        skill_indicators = ["python", "java", "javascript", "react", "node", "sql", "aws", "docker", 
                          "kubernetes", "agile", "scrum", "leadership", "management", "analysis"]
    else:
        skill_indicators = ["required", "preferred", "experience", "skills", "knowledge", "ability",
                          "bachelor", "master", "years", "minimum", "plus"]
    
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = []
    
    for word in words:
        if len(word) > 3 and word not in common_words:
            if any(indicator in word or word in indicator for indicator in skill_indicators):
                if word not in keywords:
                    keywords.append(word)
    
    return keywords[:10]

def suggest_improvements(text: str, content_type: str) -> List[str]:
    """Suggest specific improvements for the content."""
    suggestions = []
    text_lower = text.lower()
    
    if content_type == "resume":
        if "responsibility" in text_lower and "achievement" not in text_lower:
            suggestions.append("🎯 Focus on achievements rather than just responsibilities")
        
        if not re.search(r'\d+', text):
            suggestions.append("📊 Add quantifiable metrics (e.g., '20% increase', '5 years experience')")
        
        if "skill" not in text_lower:
            suggestions.append("🛠️ Include a dedicated skills section")
        
        if len(re.findall(r'\b(led|managed|created|developed|improved|achieved)\b', text_lower)) < 3:
            suggestions.append("💪 Use more action verbs (led, managed, created, developed)")
            
    else:
        if "salary" not in text_lower and "compensation" not in text_lower:
            suggestions.append("💰 Consider mentioning compensation range")
        
        if "remote" not in text_lower and "location" not in text_lower:
            suggestions.append("📍 Clarify work location/remote options")
        
        if "benefit" not in text_lower:
            suggestions.append("🎁 Highlight key benefits and perks")
        
        if not re.search(r'\d+\s*year', text_lower):
            suggestions.append("📅 Specify years of experience required")
    
    return suggestions

# ----------------------------
# AI ANALYSIS FUNCTIONS
# ----------------------------
def create_content_hash(content: str) -> str:
    """Create a hash for content to enable caching."""
    return hashlib.md5(content.encode()).hexdigest()

def extract_score_from_response(response_text: str) -> Optional[int]:
    """Extract numerical score from AI response."""
    score_patterns = [
        r'(\d+)%',
        r'score[:\s]*(\d+)',
        r'rating[:\s]*(\d+)',
        r'(\d+)/100',
        r'suitability[:\s]*(\d+)'
    ]
    
    for pattern in score_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        if matches:
            try:
                score = int(matches[0])
                return min(max(score, 0), 100)
            except ValueError:
                continue
    return None

def parse_analysis_sections(response_text: str) -> Dict[str, str]:
    """Parse AI response into structured sections."""
    sections = {
        "summary": "",
        "matches": "",
        "gaps": "",
        "recommendations": "",
        "raw_response": response_text
    }
    
    lines = response_text.split('\n')
    current_section = "summary"
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        lower_line = line.lower()
        if any(keyword in lower_line for keyword in ['match', 'strength', 'skill']):
            current_section = "matches"
        elif any(keyword in lower_line for keyword in ['gap', 'missing', 'lack', 'weak']):
            current_section = "gaps"
        elif any(keyword in lower_line for keyword in ['recommend', 'suggest', 'advice']):
            current_section = "recommendations"
        
        sections[current_section] += line + "\n"
    
    return sections

def screen_candidate(resume_text: str, job_description: str) -> Dict:
    """Enhanced candidate screening with structured analysis."""
    prompt = f"""
    As an expert HR professional and recruiter, analyze the following resume against the job description.
    
    RESUME:
    {resume_text}
    
    JOB DESCRIPTION:
    {job_description}
    
    Please provide a comprehensive analysis including:
    
    1. SUITABILITY SCORE: Provide a numerical score from 0-100% based on overall fit
    
    2. KEY MATCHES: List the specific skills, experiences, and qualifications that align well with the job requirements
    
    3. MISSING ELEMENTS: Identify important requirements from the job description that are not evident in the resume
    
    4. RECOMMENDATIONS: Suggest what the candidate could emphasize or improve to be a stronger fit
    
    5. OVERALL ASSESSMENT: Provide a summary of whether to proceed with this candidate
    
    Please be specific and provide actionable insights for both the hiring team and the candidate.
    """
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 1000
        }
    }
    
    try:
        start_time = time.time()
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            ai_response = response.json().get("response", "No analysis generated.")
            
            result = {
                "success": True,
                "response": ai_response,
                "score": extract_score_from_response(ai_response),
                "sections": parse_analysis_sections(ai_response),
                "metadata": {
                    "processing_time": round(processing_time, 2),
                    "timestamp": datetime.now().isoformat(),
                    "model": MODEL_NAME
                }
            }
            
            return result
            
        else:
            return {
                "success": False,
                "error": f"API Error (Status {response.status_code}): {response.text}",
                "score": None,
                "sections": {},
                "metadata": {"processing_time": processing_time}
            }
            
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "⏱️ Request timed out. The AI model might be processing a complex analysis.",
            "score": None,
            "sections": {},
            "metadata": {}
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "🔌 Connection error. Please ensure Ollama is running on localhost:11434",
            "score": None,
            "sections": {},
            "metadata": {}
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"❌ Unexpected error: {str(e)}",
            "score": None,
            "sections": {},
            "metadata": {}
        }

# ----------------------------
# VISUALIZATION FUNCTIONS
# ----------------------------
def create_score_visualization_plotly(score: int):
    """Create a visual score indicator using Plotly (if available)."""
    if not PLOTLY_AVAILABLE:
        return None
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Compatibility Score"},
        delta = {'reference': 70},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, font={'color': "darkblue", 'family': "Arial"})
    return fig

def create_score_visualization_html(score: int) -> str:
    """Create a visual score indicator using pure HTML/CSS."""
    if score >= 80:
        bar_class = "score-excellent"
    elif score >= 60:
        bar_class = "score-good"
    elif score >= 40:
        bar_class = "score-average"
    else:
        bar_class = "score-poor"
    
    return f"""
    <div class="score-bar-container">
        <h4 style="margin-bottom: 1rem; text-align: center;">Compatibility Score</h4>
        <div class="score-bar">
            <div class="score-fill {bar_class}" style="width: {score}%;">
                {score}%
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.9rem; color: #666;">
            <span>0%</span>
            <span>50%</span>
            <span>100%</span>
        </div>
    </div>
    """

def get_status_class(score: Optional[int]) -> str:
    """Get CSS class based on score."""
    if score is None:
        return "status-average"
    elif score >= 80:
        return "status-excellent"
    elif score >= 60:
        return "status-good"
    elif score >= 40:
        return "status-average"
    else:
        return "status-poor"

def get_status_text(score: Optional[int]) -> str:
    """Get status text based on score."""
    if score is None:
        return "Analysis Pending"
    elif score >= 80:
        return "Excellent Match"
    elif score >= 60:
        return "Good Fit"
    elif score >= 40:
        return "Moderate Fit"
    else:
        return "Poor Match"

# ----------------------------
# SIDEBAR CONFIGURATION
# ----------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Show library status
    if PYMUPDF_AVAILABLE:
        st.success(f"📄 PyMuPDF: Available ({PYMUPDF_VERSION})")
    else:
        st.error("📄 PyMuPDF: Not available")
        st.info("Install with: `pip install PyMuPDF`")
    
    if PLOTLY_AVAILABLE:
        st.success("📊 Plotly: Available")
    else:
        st.warning("📊 Plotly: Not installed")
    
    # Model settings
    with st.expander("🤖 AI Model Settings"):
        model_temp = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1, 
                              help="Controls randomness in AI responses")
        analysis_depth = st.selectbox("Analysis Depth", 
                                    ["Quick", "Standard", "Detailed"],
                                    index=1)
    
    # Analysis options
    with st.expander("📊 Analysis Options"):
        show_detailed_metrics = st.checkbox("Show Detailed Metrics", True)
        show_visualizations = st.checkbox("Show Score Visualization", True)
        use_html_charts = st.checkbox("Use HTML Charts (fallback)", not PLOTLY_AVAILABLE)
        export_results = st.checkbox("Enable Export", False)
        show_content_analysis = st.checkbox("Show Content Analysis", True)
    
    # System status
    st.markdown("### 🔍 System Status")
    status_placeholder = st.empty()
    
    # Quick stats
    st.metric("Analyses Performed", st.session_state.analysis_count)

# ----------------------------
# MAIN APPLICATION
# ----------------------------

# Header
st.markdown("""
    <div class="main-header">
        <div class="main-title">🤖 AI-Powered Job Application Screener</div>
        <div class="subtitle">Enhanced Interactive Resume Analysis & Candidate Screening</div>
    </div>
""", unsafe_allow_html=True)

# Main content columns
col1, col2 = st.columns([1, 1], gap="large")

# ----------------------------
# RESUME INPUT SECTION (NO TEMPLATES)
# ----------------------------
with col1:
    st.markdown("### 📄 Candidate Resume")
    
    # Template option completely removed
    upload_option = st.radio(
        "Input Method", 
        ["Upload PDF", "Paste Text"], 
        horizontal=True,
        help="Choose how to provide the candidate's resume"
    )
    
    resume_text = ""
    resume_metadata = {}
    
    if upload_option == "Upload PDF":
        if not PYMUPDF_AVAILABLE:
            st.error("📄 PDF processing is not available. PyMuPDF is required.")
            st.info("Install PyMuPDF with: `pip install PyMuPDF`")
            st.markdown("Please use the **Paste Text** option instead.")
        else:
            pdf_file = st.file_uploader(
                "Upload Resume PDF", 
                type=["pdf"],
                help="Upload a PDF resume for automatic text extraction"
            )
            
            if pdf_file:
                with st.spinner("🔄 Extracting text from PDF..."):
                    resume_text, resume_metadata = extract_text_from_pdf(pdf_file)
                    
                if not resume_metadata.get("error"):
                    st.success("✅ Resume successfully extracted!")
                    
                    # Show file metadata
                    if show_detailed_metrics:
                        meta_col1, meta_col2, meta_col3 = st.columns(3)
                        with meta_col1:
                            st.metric("Pages", resume_metadata.get("pages", "N/A"))
                        with meta_col2:
                            st.metric("Words", resume_metadata.get("word_count", "N/A"))
                        with meta_col3:
                            st.metric("Size (KB)", f"{resume_metadata.get('file_size', 0) // 1024}")
                    
                    # Show extraction method used
                    if "method" in resume_metadata:
                        st.caption(f"📋 Extraction method: {resume_metadata['method']}")
                    
                    # Preview option
                    if st.checkbox("👀 Preview Extracted Text"):
                        st.text_area("Extracted Text Preview", resume_text[:500] + "...", height=150, disabled=True)
                else:
                    st.error(resume_text)
                    # Show detailed error information
                    if "original_error" in resume_metadata:
                        with st.expander("🔧 Error Details"):
                            st.error(f"Original error: {resume_metadata['original_error']}")
                            if "fallback_error" in resume_metadata:
                                st.error(f"Fallback error: {resume_metadata['fallback_error']}")
    
    else:  # Paste Text
        resume_text = st.text_area(
            "Paste Resume Text", 
            height=300,
            placeholder="Paste the candidate's resume text here...",
            help="Copy and paste the resume content directly"
        )
    
    # Interactive content analysis after text input
    if resume_text and not resume_metadata.get("error"):
        # Basic metrics display
        word_count = len(resume_text.split())
        char_count = len(resume_text)
        
        # Enhanced content statistics
        st.markdown(f"""
        <div class="content-stats">
            <div class="stat-item">
                <div class="stat-value">{word_count}</div>
                <div class="stat-label">Words</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{char_count}</div>
                <div class="stat-label">Characters</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{round(word_count/200, 1)}</div>
                <div class="stat-label">Min Read</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if show_content_analysis:
            # Content quality analysis
            with st.expander("📊 Resume Quality Analysis", expanded=False):
                analysis = analyze_content_quality(resume_text, "resume")
                
                # Quality score display
                quality_class = f"quality-{analysis['quality_level']}"
                st.markdown(f"""
                <div class="quality-indicator {quality_class}">
                    Quality Score: {analysis['quality_score']}/100 ({analysis['quality_level'].title()})
                </div>
                """, unsafe_allow_html=True)
                
                # Issues and recommendations
                if analysis['issues']:
                    st.markdown("**⚠️ Issues Found:**")
                    for issue in analysis['issues']:
                        st.markdown(f"• {issue}")
                
                if analysis['recommendations']:
                    st.markdown("**💡 Recommendations:**")
                    for rec in analysis['recommendations']:
                        st.markdown(f"• {rec}")
        
        # Interactive content tools
        st.markdown("#### 🛠️ Content Enhancement Tools")
        
        tool_col1, tool_col2, tool_col3 = st.columns(3)
        
        with tool_col1:
            if st.button("🔍 Extract Keywords", key="resume_keywords"):
                keywords = extract_keywords(resume_text, "resume")
                st.markdown("**Key Skills Found:**")
                for keyword in keywords:
                    st.markdown(f"• {keyword.title()}")
        
        with tool_col2:
            if st.button("💡 Get Suggestions", key="resume_suggestions"):
                suggestions = suggest_improvements(resume_text, "resume")
                st.markdown("**Improvement Suggestions:**")
                for suggestion in suggestions:
                    st.markdown(f"• {suggestion}")
        
        with tool_col3:
            if st.button("📝 Format Check", key="resume_format"):
                # Simple formatting check
                has_bullets = "•" in resume_text or "*" in resume_text
                has_sections = len(re.findall(r'\n[A-Z][A-Z\s]+\n', resume_text)) > 2
                has_contact = any(indicator in resume_text.lower() for indicator in ["email", "@", "phone", "linkedin"])
                
                st.markdown("**Format Analysis:**")
                st.markdown(f"• Bullet points: {'✅' if has_bullets else '❌'}")
                st.markdown(f"• Clear sections: {'✅' if has_sections else '❌'}")
                st.markdown(f"• Contact info: {'✅' if has_contact else '❌'}")
        
        # Save to history
        if st.button("💾 Save to History", key="save_resume"):
            if resume_text not in [item['content'] for item in st.session_state.resume_history]:
                st.session_state.resume_history.append({
                    'content': resume_text,
                    'timestamp': datetime.now().isoformat(),
                    'word_count': word_count,
                    'source': upload_option
                })
                st.success("✅ Resume saved to history!")

# ----------------------------
# JOB DESCRIPTION SECTION (NO TEMPLATES)
# ----------------------------
with col2:
    st.markdown("### 💼 Job Description")
    
    # Template option completely removed
    job_description = st.text_area(
        "Job Requirements & Description", 
        height=350,
        placeholder="Paste the complete job description including requirements, responsibilities, and qualifications...",
        help="Provide detailed job description for accurate matching"
    )
    
    # Interactive job description analysis
    if job_description:
        # Basic metrics
        word_count = len(job_description.split())
        char_count = len(job_description)
        
        # Enhanced content statistics
        st.markdown(f"""
        <div class="content-stats">
            <div class="stat-item">
                <div class="stat-value">{word_count}</div>
                <div class="stat-label">Words</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{char_count}</div>
                <div class="stat-label">Characters</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{round(word_count/200, 1)}</div>
                <div class="stat-label">Min Read</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if show_content_analysis:
            # Job description quality analysis
            with st.expander("📊 Job Description Analysis", expanded=False):
                analysis = analyze_content_quality(job_description, "job")
                
                # Quality score display
                quality_class = f"quality-{analysis['quality_level']}"
                st.markdown(f"""
                <div class="quality-indicator {quality_class}">
                    Quality Score: {analysis['quality_score']}/100 ({analysis['quality_level'].title()})
                </div>
                """, unsafe_allow_html=True)
                
                # Issues and recommendations
                if analysis['issues']:
                    st.markdown("**⚠️ Issues Found:**")
                    for issue in analysis['issues']:
                        st.markdown(f"• {issue}")
                
                if analysis['recommendations']:
                    st.markdown("**💡 Recommendations:**")
                    for rec in analysis['recommendations']:
                        st.markdown(f"• {rec}")
        
        # Interactive job content tools
        st.markdown("#### 🛠️ Job Analysis Tools")
        
        job_tool_col1, job_tool_col2, job_tool_col3 = st.columns(3)
        
        with job_tool_col1:
            if st.button("🎯 Extract Requirements", key="job_requirements"):
                keywords = extract_keywords(job_description, "job")
                st.markdown("**Key Requirements:**")
                for keyword in keywords:
                    st.markdown(f"• {keyword.title()}")
        
        with job_tool_col2:
            if st.button("💡 Improve Description", key="job_suggestions"):
                suggestions = suggest_improvements(job_description, "job")
                st.markdown("**Enhancement Suggestions:**")
                for suggestion in suggestions:
                    st.markdown(f"• {suggestion}")
        
        with job_tool_col3:
            if st.button("📋 Check Completeness", key="job_completeness"):
                # Check for key elements
                has_responsibilities = any(word in job_description.lower() for word in ["responsibilities", "duties", "will"])
                has_requirements = any(word in job_description.lower() for word in ["requirements", "qualifications", "must"])
                has_benefits = any(word in job_description.lower() for word in ["benefits", "offer", "compensation"])
                has_company_info = any(word in job_description.lower() for word in ["company", "about us", "overview"])
                
                st.markdown("**Completeness Check:**")
                st.markdown(f"• Responsibilities: {'✅' if has_responsibilities else '❌'}")
                st.markdown(f"• Requirements: {'✅' if has_requirements else '❌'}")
                st.markdown(f"• Benefits: {'✅' if has_benefits else '❌'}")
                st.markdown(f"• Company Info: {'✅' if has_company_info else '❌'}")
        
        # Save to history
        if st.button("💾 Save to History", key="save_job"):
            if job_description not in [item['content'] for item in st.session_state.job_history]:
                st.session_state.job_history.append({
                    'content': job_description,
                    'timestamp': datetime.now().isoformat(),
                    'word_count': word_count,
                    'source': "paste_text"
                })
                st.success("✅ Job description saved to history!")

# ----------------------------
# HISTORY MANAGEMENT (KEPT)
# ----------------------------
if st.session_state.resume_history or st.session_state.job_history:
    st.markdown("---")
    st.markdown("### 📚 Content History")
    
    history_col1, history_col2 = st.columns(2)
    
    with history_col1:
        if st.session_state.resume_history:
            st.markdown("#### 📄 Resume History")
            for i, item in enumerate(st.session_state.resume_history[-3:]):  # Show last 3
                timestamp = datetime.fromisoformat(item['timestamp']).strftime("%m/%d %H:%M")
                if st.button(f"Resume {len(st.session_state.resume_history)-i} ({item['word_count']} words) - {timestamp}", 
                           key=f"load_resume_{i}"):
                    resume_text = item['content']
                    st.success("✅ Resume loaded from history!")
                    st.rerun()
    
    with history_col2:
        if st.session_state.job_history:
            st.markdown("#### 💼 Job History")
            for i, item in enumerate(st.session_state.job_history[-3:]):  # Show last 3
                timestamp = datetime.fromisoformat(item['timestamp']).strftime("%m/%d %H:%M")
                if st.button(f"Job {len(st.session_state.job_history)-i} ({item['word_count']} words) - {timestamp}", 
                           key=f"load_job_{i}"):
                    job_description = item['content']
                    st.success("✅ Job description loaded from history!")
                    st.rerun()

# ----------------------------
# ANALYSIS SECTION (UNCHANGED)
# ----------------------------
st.markdown("---")

# Analysis button
analysis_ready = bool(resume_text.strip() and job_description.strip() and not resume_metadata.get("error"))

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze_button = st.button(
        "🔍 Analyze Candidate Compatibility",
        use_container_width=True,
        disabled=not analysis_ready,
        type="primary"
    )

if not analysis_ready:
    if resume_metadata.get("error"):
        st.error("📄 Please fix the PDF processing error before proceeding with analysis.")
    else:
        st.info("📋 Please provide both a resume and job description to begin analysis.")

# Perform analysis
if analyze_button and analysis_ready:
    with status_placeholder:
        st.success("🔄 Processing Analysis...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, step in enumerate(["Parsing resume...", "Analyzing job requirements...", "Comparing compatibility...", "Generating insights..."]):
        status_text.text(step)
        progress_bar.progress((i + 1) / 4)
        time.sleep(0.5)
    
    with st.spinner("🤖 AI is analyzing the candidate..."):
        result = screen_candidate(resume_text, job_description)
        st.session_state.analysis_count += 1
    
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    if result["success"]:
        st.markdown("## 📊 Analysis Results")
        
        # Score display
        score = result.get("score")
        if score is not None and show_visualizations:
            col_score1, col_score2 = st.columns([1, 2])
            
            with col_score1:
                st.markdown(f"""
                    <div class="score-container">
                        <div class="score-title">Compatibility Score</div>
                        <div class="score-value">{score}%</div>
                    </div>
                """, unsafe_allow_html=True)
                
                status_class = get_status_class(score)
                status_text = get_status_text(score)
                st.markdown(f'<div class="status-indicator {status_class}">{status_text}</div>', 
                           unsafe_allow_html=True)
            
            with col_score2:
                if PLOTLY_AVAILABLE and not use_html_charts:
                    fig = create_score_visualization_plotly(score)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    html_chart = create_score_visualization_html(score)
                    st.markdown(html_chart, unsafe_allow_html=True)
        
        # Detailed analysis tabs
        sections = result.get("sections", {})
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "✅ Matches", "❌ Gaps", "💡 Recommendations"])
        
        with tab1:
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            summary = sections.get("summary", result["response"][:500] + "...")
            st.write(summary)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            matches = sections.get("matches", "Analyzing skill matches...")
            st.write(matches if matches.strip() else "Processing match analysis...")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            gaps = sections.get("gaps", "Identifying skill gaps...")
            st.write(gaps if gaps.strip() else "Processing gap analysis...")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            recommendations = sections.get("recommendations", "Generating recommendations...")
            st.write(recommendations if recommendations.strip() else "Processing recommendations...")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Full analysis (expandable)
        with st.expander("🔍 View Complete Analysis"):
            st.text_area("Full AI Response", result["response"], height=300, disabled=True)
        
        # Metadata
        if show_detailed_metrics:
            st.markdown("### 📈 Analysis Metadata")
            metadata = result.get("metadata", {})
            
            met_col1, met_col2, met_col3 = st.columns(3)
            with met_col1:
                st.metric("Processing Time", f"{metadata.get('processing_time', 'N/A')}s")
            with met_col2:
                st.metric("Model Used", metadata.get('model', MODEL_NAME))
            with met_col3:
                st.metric("Analysis Date", 
                         datetime.fromisoformat(metadata.get('timestamp', datetime.now().isoformat())).strftime("%Y-%m-%d %H:%M") 
                         if metadata.get('timestamp') else 'N/A')
        
        # Export options
        if export_results:
            st.markdown("### 📤 Export Results")
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                export_data = {
                    "candidate_analysis": result,
                    "resume_metadata": resume_metadata,
                    "analysis_timestamp": datetime.now().isoformat()
                }
                st.download_button(
                    "📄 Download as JSON",
                    json.dumps(export_data, indent=2),
                    file_name=f"candidate_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with export_col2:
                report = f"""
CANDIDATE ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Compatibility Score: {score}%
Status: {get_status_text(score)}

ANALYSIS:
{result['response']}
                """
                st.download_button(
                    "📝 Download as Text Report",
                    report,
                    file_name=f"candidate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    else:
        # Error handling
        st.error("❌ Analysis Failed")
        st.error(result.get("error", "Unknown error occurred"))
        
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **Common issues and solutions:**
            
            1. **Connection Error**: Ensure Ollama is running on localhost:11434
            2. **Timeout Error**: The analysis might be complex. Try with shorter text.
            3. **Model Error**: Verify that the Mistral-Nemo model is downloaded in Ollama
            4. **Memory Error**: Close other applications and try again
            
            **Quick fixes:**
            - Restart Ollama: `ollama serve`
            - Check model: `ollama list`
            - Pull model: `ollama pull mistral-nemo:latest`
            """)

# Update sidebar status
with status_placeholder:
    if 'result' in locals() and result["success"]:
        st.success("✅ Analysis Complete")
    elif 'result' in locals():
        st.error("❌ Analysis Failed")
    else:
        st.info("⏳ Ready for Analysis")

# ----------------------------
# ENHANCED FOOTER
# ----------------------------
st.markdown(f"""
    <div class="footer">
        <p>🚀 <strong>AI Job Application Screener v3.1</strong> | 
        Fixed PyMuPDF Integration | 
        Powered by <strong>Mistral-Nemo</strong> LLM via Ollama</p>
        <p>💡 <em>Professional hiring solution with enhanced PDF processing and smart analysis tools</em></p>
    </div>
""", unsafe_allow_html=True)