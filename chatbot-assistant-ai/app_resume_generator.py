"""
AI Resume Builder - Conversational Assistant
An interactive AI assistant that helps users build professional, ATS-optimized resumes
through conversational dialogue and provides formatting guidance.
"""

import os
import re
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv

import streamlit as st
import openai

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Load environment variables
load_dotenv()

# ============================
# Configuration
# ============================

st.set_page_config(
    page_title="AI Resume Builder",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Model configurations
OLLAMA_MODELS = {
    "gpt-oss:120b-cloud": {"name": "GPT-OSS 120B (Cloud)", "provider": "ollama_cloud"},
    "gpt-oss:20b-cloud": {"name": "GPT-OSS 20B (Cloud)", "provider": "ollama_cloud"},
    "deepseek-v3.1:671b-cloud": {"name": "DeepSeek V3.1 671B (Cloud)", "provider": "ollama_cloud"},
    "qwen3-coder:480b-cloud": {"name": "Qwen3 Coder 480B (Cloud)", "provider": "ollama_cloud"},
    "kimi-k2:1t-cloud": {"name": "Kimi K2 1T (Cloud)", "provider": "ollama_cloud"},
    "minimax-m2:cloud": {"name": "MiniMax M2 (Cloud)", "provider": "ollama_cloud"},
    "glm-4.6:cloud": {"name": "GLM 4.6 (Cloud)", "provider": "ollama_cloud"},
    "gpt-oss:20b": {"name": "GPT-OSS 20B (Local)", "provider": "ollama"},
    "mistral-nemo:latest": {"name": "Mistral Nemo (Local)", "provider": "ollama"},
    "codellama:13b": {"name": "CodeLlama 13B (Local)", "provider": "ollama"},
    "llama3.1:8b": {"name": "Llama 3.1 8B (Local)", "provider": "ollama"},
}

OPENAI_MODELS = {
    "gpt-5-nano": {"name": "GPT-5 Nano", "provider": "openai"},
    "gpt-4.1-nano": {"name": "GPT-4.1 Nano", "provider": "openai"},
}

DEFAULT_MODEL = "gpt-oss:120b-cloud"

# ============================
# Session State Initialization
# ============================

def initialize_session_state():
    """Initialize all session state variables"""
    if "history" not in st.session_state:
        st.session_state.history = []

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = DEFAULT_MODEL

    if "model_provider" not in st.session_state:
        st.session_state.model_provider = "ollama_cloud"

    if "input_key" not in st.session_state:
        st.session_state.input_key = 0

    if "theme" not in st.session_state:
        st.session_state.theme = "default"

    if "show_timestamps" not in st.session_state:
        st.session_state.show_timestamps = False

    if "resume_mode" not in st.session_state:
        st.session_state.resume_mode = "Professional"

initialize_session_state()


# ============================
# Model Provider Functions
# ============================

def clean_html_tags(text: str) -> str:
    """Remove all HTML tags from text"""
    cleaned = re.sub(r'<[^>]+>', '', text)
    return cleaned


def call_ollama(messages: List[Dict], model: str) -> str:
    """Call Ollama API"""
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
        )
        content = response["message"]["content"].strip()
        return clean_html_tags(content)
    except Exception as e:
        return f"Ollama Error: {str(e)}"


def call_openai(messages: List[Dict], model: str) -> str:
    """Call OpenAI API"""
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return "Error: OpenAI API key not found in environment variables."

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )

        content = response.choices[0].message.content.strip() if response.choices else ""

        if not content:
            return "I'm having trouble generating that section. Could you provide more details?"

        return clean_html_tags(content)

    except Exception as e:
        return f"OpenAI Error: {str(e)}"


# ============================
# Main Chat Function
# ============================

def get_resume_response(user_input: str) -> str:
    """
    Get resume building assistance from selected AI model
    """
    resume_modes = {
        "Professional": (
            "You are an expert resume writer and career coach. Help users build professional, "
            "ATS-optimized resumes through conversational guidance. Ask clarifying questions about their "
            "experience, skills, education, and career goals. Provide specific advice on how to phrase "
            "accomplishments using action verbs and quantifiable metrics. Suggest improvements to make "
            "their resume more impactful and industry-appropriate. When they provide information, help "
            "them craft compelling bullet points and sections. Be encouraging and constructive. "
            "Format your responses using plain text with markdown. Never use HTML tags like <br>, <b>, <i>, etc. "
            "Use markdown syntax instead (line breaks, **bold**, *italic*)."
        ),
        "Resume Enhancer": (
            "You are a professional resume enhancement specialist who transforms plain or basic resume bullet points "
            "into powerful, impactful statements that make job seekers stand out. Your expertise lies in rewriting "
            "resume content to be more professional, action-driven, and achievement-oriented. When users provide "
            "their existing bullet points or resume sections, you rewrite them using: strong action verbs, "
            "quantifiable achievements (metrics, percentages, numbers), industry-specific keywords for ATS optimization, "
            "clear cause-and-effect relationships showing impact, and professional language that highlights value. "
            "You also identify weak phrases like 'responsible for' or 'helped with' and transform them into powerful "
            "statements that demonstrate concrete accomplishments. Ask clarifying questions about scope, impact, "
            "or metrics if the original content lacks specificity. Then provide multiple enhanced versions with "
            "explanations of what makes each version stronger. Be enthusiastic about helping job seekers level up "
            "their resume game and stand out in competitive job markets. "
            "Format your responses using plain text with markdown. Never use HTML tags like <br>, <b>, <i>, etc. "
            "Use markdown syntax instead (line breaks, **bold**, *italic*)."
        ),
        "Technical": (
            "You are an expert technical resume writer specializing in engineering and tech roles. "
            "Help users create resumes that highlight technical skills, projects, certifications, and "
            "quantifiable achievements. Focus on programming languages, frameworks, tools, system design, "
            "and technical methodologies. Guide them to use industry-standard keywords for ATS optimization. "
            "Ask about their tech stack, project impact, and technical leadership experience. Provide examples "
            "of strong technical bullet points with metrics and concrete results. "
            "Format your responses using plain text with markdown. Never use HTML tags like <br>, <b>, <i>, etc. "
            "Use markdown syntax instead (line breaks, **bold**, *italic*)."
        ),
        "Creative": (
            "You are an expert resume writer for creative professionals in design, marketing, and media. "
            "Help users showcase their creative work, campaigns, portfolios, and innovative projects. "
            "Focus on impact metrics, audience reach, brand development, and artistic achievements. "
            "Guide them to balance creativity with professionalism and ATS-friendliness. Ask about their "
            "creative process, tools, and measurable results. Emphasize visual and conceptual accomplishments. "
            "Format your responses using plain text with markdown. Never use HTML tags like <br>, <b>, <i>, etc. "
            "Use markdown syntax instead (line breaks, **bold**, *italic*)."
        ),
        "Executive": (
            "You are an executive resume writer specializing in senior leadership roles. Help users "
            "create strategic, results-driven resumes that highlight leadership experience, strategic vision, "
            "P&L responsibility, organizational transformation, and business impact. Focus on high-level "
            "accomplishments, team leadership, revenue growth, operational excellence, and executive presence. "
            "Ask about their leadership philosophy, team size, budget responsibility, and strategic initiatives. "
            "Format your responses using plain text with markdown. Never use HTML tags like <br>, <b>, <i>, etc. "
            "Use markdown syntax instead (line breaks, **bold**, *italic*)."
        ),
        "Entry-Level": (
            "You are a resume writer specializing in entry-level and early-career professionals. "
            "Help users create strong resumes even with limited work experience by emphasizing education, "
            "internships, projects, relevant coursework, volunteer work, and transferable skills. Guide them "
            "to showcase potential, enthusiasm, and quick learning ability. Ask about academic achievements, "
            "extracurriculars, personal projects, and any work experience. Help them frame experiences "
            "professionally and identify transferable skills. "
            "Format your responses using plain text with markdown. Never use HTML tags like <br>, <b>, <i>, etc. "
            "Use markdown syntax instead (line breaks, **bold**, *italic*)."
        ),
    }

    system_message = {
        "role": "system",
        "content": resume_modes.get(st.session_state.resume_mode, resume_modes["Professional"]),
    }

    messages = [system_message]

    for msg in st.session_state.history:
        if msg["role"] in ("user", "assistant"):
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

    messages.append({"role": "user", "content": user_input})

    provider = st.session_state.model_provider
    model = st.session_state.selected_model

    if provider == "openai":
        return call_openai(messages, model)
    elif provider in ("ollama", "ollama_cloud"):
        return call_ollama(messages, model)
    else:
        return "Error: Unknown model provider selected."


# ============================
# UI Components
# ============================

def get_theme_colors():
    """Return theme-specific color schemes"""
    themes = {
        "default": {
            "user_bg": "#e3f2fd",
            "bot_bg": "#f5f5f5",
            "gradient_start": "#2196f3",
            "gradient_end": "#64b5f6",
        },
        "dark": {
            "user_bg": "#263238",
            "bot_bg": "#37474f",
            "gradient_start": "#1565c0",
            "gradient_end": "#1976d2",
        },
        "professional": {
            "user_bg": "#e8eaf6",
            "bot_bg": "#fafafa",
            "gradient_start": "#3f51b5",
            "gradient_end": "#5c6bc0",
        },
        "modern": {
            "user_bg": "#e0f2f1",
            "bot_bg": "#fafafa",
            "gradient_start": "#00897b",
            "gradient_end": "#26a69a",
        },
        "enhancer": {
            "user_bg": "#fff3e0",
            "bot_bg": "#fafafa",
            "gradient_start": "#ff6f00",
            "gradient_end": "#ffa726",
        },
    }
    return themes.get(st.session_state.theme, themes["default"])


def display_chat_history():
    """Display chat history with modern UI"""
    if not st.session_state.history:
        st.info("👋 Hi! I'm your AI Resume Builder. Let's create an amazing resume together! Tell me about your target role, experience, skills, or ask for guidance.")
        return

    for msg in st.session_state.history:
        timestamp_display = ""
        if st.session_state.show_timestamps and "timestamp" in msg:
            try:
                dt = datetime.fromisoformat(msg["timestamp"])
                timestamp_display = f" {dt.strftime('%H:%M')}"
            except Exception:
                pass

        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"**You{timestamp_display}**")
                st.write(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(f"**Resume Builder{timestamp_display}**")
                st.write(msg["content"])


def render_sidebar():
    """Render sidebar with settings"""
    with st.sidebar:
        st.title("Settings")

        st.divider()

        # Resume Mode Selection
        st.subheader("Resume Style")

        resume_modes = ["Professional", "Resume Enhancer", "Technical", "Creative", "Executive", "Entry-Level"]
        st.session_state.resume_mode = st.selectbox(
            "Mode",
            options=resume_modes,
            index=resume_modes.index(st.session_state.resume_mode),
            help="Choose your resume specialization",
        )

        st.divider()

        # Model Provider Selection
        st.subheader("Model Selection")

        provider_options = {
            "Ollama (Cloud)": "ollama_cloud",
            "Ollama (Local)": "ollama",
            "OpenAI": "openai",
        }

        selected_provider_name = st.selectbox(
            "Provider",
            options=list(provider_options.keys()),
            index=0,
            help="Choose your AI model provider",
        )

        selected_provider = provider_options[selected_provider_name]

        if selected_provider == "openai":
            available_models = {k: v["name"] for k, v in OPENAI_MODELS.items()}
            model_key = st.selectbox(
                "Model",
                options=list(available_models.keys()),
                format_func=lambda x: available_models[x],
                help="Select OpenAI model",
            )
            st.session_state.model_provider = "openai"
            st.session_state.selected_model = model_key

            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                st.success("API Key loaded from .env")
            else:
                st.error("No API key found in .env")

        elif selected_provider == "ollama":
            local_models = {k: v["name"] for k, v in OLLAMA_MODELS.items() if v["provider"] == "ollama"}
            if local_models:
                model_key = st.selectbox(
                    "Model",
                    options=list(local_models.keys()),
                    format_func=lambda x: local_models[x],
                    help="Select local Ollama model",
                )
                st.session_state.model_provider = "ollama"
                st.session_state.selected_model = model_key
            else:
                st.warning("No local Ollama models found")

        else:  # ollama_cloud
            cloud_models = {k: v["name"] for k, v in OLLAMA_MODELS.items() if v["provider"] == "ollama_cloud"}
            model_key = st.selectbox(
                "Model",
                options=list(cloud_models.keys()),
                format_func=lambda x: cloud_models[x],
                index=0,
                help="Select Ollama cloud model",
            )
            st.session_state.model_provider = "ollama_cloud"
            st.session_state.selected_model = model_key

        st.info(f"**Current Model:**\n{st.session_state.selected_model}")

        st.divider()

        # UI Customization
        st.subheader("Appearance")

        theme_options = ["default", "dark", "professional", "modern", "enhancer"]
        st.session_state.theme = st.selectbox(
            "Theme",
            options=theme_options,
            index=theme_options.index(st.session_state.theme),
            format_func=lambda x: x.title(),
        )

        st.session_state.show_timestamps = st.checkbox(
            "Show timestamps",
            value=st.session_state.show_timestamps,
        )

        st.divider()

        # Quick Actions
        st.subheader("Quick Actions")

        if st.button("Sample Question", use_container_width=True):
            sample_prompts = {
                "Professional": "I'm applying for a Project Manager role. Help me write a strong professional summary.",
                "Resume Enhancer": "Can you enhance this bullet point: 'Responsible for managing customer accounts and handling support tickets.'",
                "Technical": "I'm a software engineer with 5 years of experience in Python and cloud technologies. How should I structure my technical skills section?",
                "Creative": "I'm a graphic designer. How can I showcase my design projects effectively on my resume?",
                "Executive": "I'm a VP of Operations. How do I highlight my strategic leadership and P&L responsibility?",
                "Entry-Level": "I'm a recent graduate looking for my first job. How do I make my resume stand out with limited experience?",
            }
            prompt = sample_prompts.get(st.session_state.resume_mode, sample_prompts["Professional"])
            st.session_state.history.append({
                "role": "assistant",
                "content": f"Here's a sample question to get started:\n\n{prompt}\n\nFeel free to share your own details, and I'll help you craft a great resume!",
                "timestamp": datetime.now().isoformat(),
            })
            st.rerun()

        st.divider()

        # Session Controls
        st.subheader("Session Controls")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("New Session", use_container_width=True):
                st.session_state.history = []
                st.session_state.input_key += 1
                st.rerun()

        with col2:
            if st.button("Export", use_container_width=True):
                if st.session_state.history:
                    export_chat()

        if st.session_state.history:
            st.divider()
            st.subheader("Statistics")

            total_messages = len(st.session_state.history)
            user_messages = len([m for m in st.session_state.history if m["role"] == "user"])
            bot_messages = total_messages - user_messages

            col1, col2 = st.columns(2)
            col1.metric("Your inputs", user_messages)
            col2.metric("AI responses", bot_messages)

        st.divider()

        # Resume Tips
        st.subheader("Resume Tips")
        st.info(
            "**ATS Optimization:**\n"
            "- Use standard section headers\n"
            "- Include relevant keywords\n"
            "- Use bullet points with metrics\n"
            "- Keep formatting simple and clean"
        )


def export_chat():
    """Export chat history to text file"""
    if not st.session_state.history:
        st.error("No conversation to export")
        return

    export_text = "AI Resume Builder Session Export\n"
    export_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_text += f"Mode: {st.session_state.resume_mode}\n"
    export_text += f"Model: {st.session_state.selected_model}\n"
    export_text += "=" * 50 + "\n\n"

    for msg in st.session_state.history:
        role = "You" if msg["role"] == "user" else "Resume Builder"
        timestamp = msg.get("timestamp", "")
        export_text += f"[{timestamp}] {role}:\n{msg['content']}\n\n"

    st.sidebar.download_button(
        label="Download Session",
        data=export_text,
        file_name=f"resume_builder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ============================
# Main Application
# ============================

def main():
    """Main application logic"""

    render_sidebar()

    _, col2, _ = st.columns([1, 6, 1])

    with col2:
        st.markdown(
            """
            <div style='text-align: center; padding: 20px;'>
                <h1 style='color: #2196f3; font-size: 3em; margin-bottom: 0;'>📄 AI Resume Builder</h1>
                <p style='color: #666; font-size: 1.2em;'>Build professional, ATS-optimized resumes with AI assistance</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #2196f3 0%, #64b5f6 100%);
            padding: 20px; border-radius: 15px; color: white; text-align: center; margin-bottom: 30px;'>
                <p style='margin: 0; font-size: 1.1em;'>
                    <strong>Create a standout resume!</strong><br/>
                    Get expert guidance on structuring, writing, and optimizing your resume for any role.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        display_chat_history()

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Message",
                placeholder="Tell me about your experience, skills, or ask for resume advice...",
                key=f"user_input_{st.session_state.input_key}",
                label_visibility="collapsed",
                height=100,
                max_chars=2000,
            )

            submitted = st.form_submit_button("Send", use_container_width=True, type="primary")

        if submitted and user_input.strip():
            st.session_state.history.append({
                "role": "user",
                "content": user_input.strip(),
                "timestamp": datetime.now().isoformat(),
            })

            with st.spinner("Building your resume..."):
                reply = get_resume_response(user_input.strip())

            st.session_state.history.append({
                "role": "assistant",
                "content": reply,
                "timestamp": datetime.now().isoformat(),
            })

            st.session_state.input_key += 1
            st.rerun()


# ============================
# Custom CSS
# ============================

def apply_custom_css():
    """Apply custom CSS styling"""
    colors = get_theme_colors()

    st.markdown(
        f"""
        <style>
        .main .block-container {{
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }}
        .stTextInput > div > div > input {{
            border-radius: 25px;
            border: 2px solid #e0e0e0;
            padding: 12px 20px;
            font-size: 1em;
            transition: all 0.3s ease;
        }}
        .stTextInput > div > div > input:focus {{
            border-color: {colors["gradient_start"]};
            box-shadow: 0 0 15px rgba(33, 150, 243, 0.3);
        }}
        .stTextArea > div > div > textarea {{
            border-radius: 15px;
            border: 2px solid #e0e0e0;
            padding: 12px 20px;
            font-size: 1em;
            transition: all 0.3s ease;
            resize: vertical;
        }}
        .stTextArea > div > div > textarea:focus {{
            border-color: {colors["gradient_start"]};
            box-shadow: 0 0 15px rgba(33, 150, 243, 0.3);
        }}
        .stButton > button, .stFormSubmitButton > button {{
            border-radius: 25px;
            border: none;
            background: linear-gradient(90deg, {colors["gradient_start"]} 0%, {colors["gradient_end"]} 100%);
            color: white;
            font-weight: 600;
            padding: 12px 24px;
            transition: all 0.3s ease;
            font-size: 1em;
        }}
        .stButton > button:hover, .stFormSubmitButton > button:hover {{
            background: linear-gradient(90deg, {colors["gradient_end"]} 0%, {colors["gradient_start"]} 100%);
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(33, 150, 243, 0.4);
        }}
        .stSelectbox > div > div {{
            border-radius: 10px;
            border: 2px solid #e0e0e0;
        }}
        .css-1d391kg {{
            padding-top: 2rem;
        }}
        ::-webkit-scrollbar {{
            width: 10px;
        }}
        ::-webkit-scrollbar-track {{
            background: #f1f1f1;
            border-radius: 10px;
        }}
        ::-webkit-scrollbar-thumb {{
            background: linear-gradient(180deg, {colors["gradient_start"]} 0%, {colors["gradient_end"]} 100%);
            border-radius: 10px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: {colors["gradient_end"]};
        }}
        .stForm {{
            border: none;
            padding: 20px 0;
        }}
        .css-1xarl3l {{
            background: linear-gradient(135deg, {colors["gradient_start"]}22 0%, {colors["gradient_end"]}22 100%);
            border-radius: 10px;
            padding: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================
# Run Application
# ============================

if __name__ == "__main__":
    apply_custom_css()
    main()
