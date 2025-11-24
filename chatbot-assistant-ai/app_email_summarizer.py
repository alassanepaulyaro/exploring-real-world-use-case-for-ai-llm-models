# Email Assistant
"""
This tool summarizes long, back-and-forth email threads into a concise overview—
highlighting who said what, key decisions, and action items. It also rewrites emails
professionally and corrects grammar, spelling, and punctuation for polished writing.
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
    page_title="Email Assistant",
    page_icon="📨",
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

    if "summary_focus" not in st.session_state:
        st.session_state.summary_focus = "Balanced"

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
            return "I couldn't generate a summary right now. Could you try again?"

        return clean_html_tags(content)

    except Exception as e:
        return f"OpenAI Error: {str(e)}"


# ============================
# Main Summarization Function
# ============================

def get_email_summary(user_input: str) -> str:
    """
    Get email thread summary from selected AI model
    """
    summary_focuses = {
        "Balanced": (
            "You are an executive assistant. Your job is to read an entire email thread, "
            "highlight key points, decisions made, who said what, and any action items. "
            "Be clear, concise, and professional. Use bullet points if helpful. "
            "Format your responses using plain text with markdown. Never use HTML tags like <br>, <b>, <i>, etc. "
            "Use markdown syntax instead (line breaks, **bold**, *italic*)."
        ),
        "Email Rewriter": (
            "You are a professional email writing expert. Your specialty is rewriting emails to improve their tone, "
            "clarity, and effectiveness. You can adapt any email into a more professional, polite, concise, or friendly tone—"
            "depending on what the user needs. You're perfect for job applications, cold outreach, internal communication, "
            "or simply leveling up someone's email game. When given an email, ask about the desired tone if not specified "
            "(professional, friendly, concise, polite, formal, casual, persuasive), then rewrite the email accordingly. "
            "Maintain the core message while improving structure, word choice, grammar, and overall impact. "
            "Provide the rewritten version clearly and explain key improvements made. "
            "Format your responses using plain text with markdown. Never use HTML tags like <br>, <b>, <i>, etc. "
            "Use markdown syntax instead (line breaks, **bold**, *italic*)."
        ),
        "Grammar Corrector": (
            "You are an expert grammar and writing assistant who corrects spelling, grammar, and punctuation mistakes "
            "in any text. Your specialty is polishing writing to make it professional, clear, and easy to read. "
            "When users provide text (emails, essays, social posts, documents), you identify and fix errors including: "
            "spelling mistakes, grammatical errors, punctuation problems, awkward phrasing, run-on sentences, "
            "subject-verb agreement issues, and unclear wording. You also improve clarity, flow, and readability "
            "while preserving the author's original voice and intent. Present the corrected version first, then "
            "briefly explain the main corrections you made. Be thorough but maintain the natural style of the writing. "
            "Perfect for anyone who wants polished, error-free writing for professional or personal use. "
            "Format your responses using plain text with markdown. Never use HTML tags like <br>, <b>, <i>, etc. "
            "Use markdown syntax instead (line breaks, **bold**, *italic*)."
        ),
        "Action Items": (
            "You are an executive assistant specializing in action item extraction. "
            "Read the email thread and identify all action items, tasks, and deliverables. "
            "For each action item, specify who is responsible, what needs to be done, and any mentioned deadlines. "
            "Prioritize clarity and actionability. "
            "Format your responses using plain text with markdown. Never use HTML tags like <br>, <b>, <i>, etc. "
            "Use markdown syntax instead (line breaks, **bold**, *italic*)."
        ),
        "Decisions": (
            "You are an executive assistant specializing in decision tracking. "
            "Read the email thread and extract all decisions that were made, who made them, "
            "the rationale provided, and any implications or next steps. "
            "Organize by importance and chronological order. "
            "Format your responses using plain text with markdown. Never use HTML tags like <br>, <b>, <i>, etc. "
            "Use markdown syntax instead (line breaks, **bold**, *italic*)."
        ),
        "Key Participants": (
            "You are an executive assistant specializing in conversation analysis. "
            "Read the email thread and summarize the conversation by participant. "
            "For each key participant, highlight their main points, requests, concerns, and contributions. "
            "This helps understand different perspectives in the thread. "
            "Format your responses using plain text with markdown. Never use HTML tags like <br>, <b>, <i>, etc. "
            "Use markdown syntax instead (line breaks, **bold**, *italic*)."
        ),
    }

    system_message = {
        "role": "system",
        "content": summary_focuses.get(st.session_state.summary_focus, summary_focuses["Balanced"]),
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
            "gradient_start": "#1565c0",
            "gradient_end": "#42a5f5",
        },
        "dark": {
            "user_bg": "#132743",
            "bot_bg": "#1f4068",
            "gradient_start": "#0d47a1",
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
        "writer": {
            "user_bg": "#fce4ec",
            "bot_bg": "#fafafa",
            "gradient_start": "#c2185b",
            "gradient_end": "#e91e63",
        },
        "corrector": {
            "user_bg": "#f3e5f5",
            "bot_bg": "#fafafa",
            "gradient_start": "#7b1fa2",
            "gradient_end": "#ab47bc",
        },
    }
    return themes.get(st.session_state.theme, themes["default"])


def display_chat_history():
    """Display chat history with modern UI"""
    if not st.session_state.history:
        st.info("📨 Paste an email to summarize, rewrite, correct grammar, or analyze! Choose your tool from the sidebar.")
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
                st.markdown(f"**Summarizer{timestamp_display}**")
                st.write(msg["content"])


def render_sidebar():
    """Render sidebar with settings"""
    with st.sidebar:
        st.title("Settings")

        st.divider()

        # Summary Focus Selection
        st.subheader("Email Tools")

        summary_focuses = ["Balanced", "Email Rewriter", "Grammar Corrector", "Action Items", "Decisions", "Key Participants"]
        st.session_state.summary_focus = st.selectbox(
            "Tool",
            options=summary_focuses,
            index=summary_focuses.index(st.session_state.summary_focus),
            help="Choose the email tool you need",
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
            col1.metric("Email threads", user_messages)
            col2.metric("Summaries", bot_messages)


def export_chat():
    """Export chat history to text file"""
    if not st.session_state.history:
        st.error("No conversation to export")
        return

    export_text = "Email Assistant Export\n"
    export_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_text += f"Focus: {st.session_state.summary_focus}\n"
    export_text += f"Model: {st.session_state.selected_model}\n"
    export_text += "=" * 50 + "\n\n"

    for msg in st.session_state.history:
        role = "Email Thread" if msg["role"] == "user" else "Summary"
        timestamp = msg.get("timestamp", "")
        export_text += f"[{timestamp}] {role}:\n{msg['content']}\n\n"

    st.sidebar.download_button(
        label="Download Session",
        data=export_text,
        file_name=f"email_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
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
                <h1 style='color: #1565c0; font-size: 3em; margin-bottom: 0;'>📨 Email Assistant</h1>
                <p style='color: #666; font-size: 1.2em;'>Summarize, rewrite, and analyze emails with AI</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #1565c0 0%, #42a5f5 100%);
            padding: 20px; border-radius: 15px; color: white; text-align: center; margin-bottom: 30px;'>
                <p style='margin: 0; font-size: 1.1em;'>
                    <strong>Master your email communication!</strong><br/>
                    Summarize threads, rewrite messages professionally, correct grammar, extract action items, and more.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        display_chat_history()

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Message",
                placeholder="Paste an email or thread here...",
                key=f"user_input_{st.session_state.input_key}",
                label_visibility="collapsed",
                height=200,
                max_chars=10000,
            )

            submitted = st.form_submit_button("Summarize", use_container_width=True, type="primary")

        if submitted and user_input.strip():
            st.session_state.history.append({
                "role": "user",
                "content": user_input.strip(),
                "timestamp": datetime.now().isoformat(),
            })

            with st.spinner("Generating summary..."):
                reply = get_email_summary(user_input.strip())

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
            box-shadow: 0 0 15px rgba(21, 101, 192, 0.3);
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
            box-shadow: 0 0 15px rgba(21, 101, 192, 0.3);
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
            box-shadow: 0 5px 20px rgba(21, 101, 192, 0.4);
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
