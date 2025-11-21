# Daily Productivity Planner
"""
Transforms tasks, meetings, and energy levels into realistic daily schedules.
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
    page_title="Daily Productivity Planner",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

    if "planner_mode" not in st.session_state:
        st.session_state.planner_mode = "Balanced Day"


initialize_session_state()


# ============================
# Model Provider Functions
# ============================


def clean_html_tags(text: str) -> str:
    """Strip HTML tags"""
    return re.sub(r"<[^>]+>", "", text)


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
            return "I could not outline your day yet - could you add a bit more context?"

        return clean_html_tags(content)

    except Exception as e:
        return f"OpenAI Error: {str(e)}"


# ============================
# Main Chat Function
# ============================


def get_planner_response(user_input: str) -> str:
    """Route conversation through the selected AI model"""
    planner_modes = {
        "Balanced Day": (
            "You are a productivity coach building balanced daily plans. "
            "Create chronological schedules with focus blocks, breaks, meals, and habit cues. "
            "Reference energy management and highlight top priorities. Use markdown."
        ),
        "Deep Work Sprint": (
            "You are a workflow strategist optimizing for long focus sessions. "
            "Batch meetings, cluster errands, and insert recovery moments. "
            "Offer micro-motivation tips for each block. Use markdown."
        ),
        "Meeting Heavy": (
            "You are a professional assistant organizing a meeting-packed day. "
            "Include prep buffers, follow-up slots, and quick recharge suggestions. Use markdown."
        ),
        "Student Mode": (
            "You are an academic mentor helping learners juggle classes, study, and life. "
            "Include study intervals, review sessions, and accountability checkpoints. Use markdown."
        ),
    }

    system_message = {
        "role": "system",
        "content": planner_modes.get(st.session_state.planner_mode, planner_modes["Balanced Day"]),
    }

    messages = [system_message]

    for msg in st.session_state.history:
        if msg["role"] in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_input})

    provider = st.session_state.model_provider
    model = st.session_state.selected_model

    if provider == "openai":
        return call_openai(messages, model)
    if provider in ("ollama", "ollama_cloud"):
        return call_ollama(messages, model)
    return "Error: Unknown model provider selected."


# ============================
# UI Components
# ============================


def get_theme_colors():
    """Return theme-specific color schemes"""
    themes = {
        "default": {
            "user_bg": "#fff3e0",
            "bot_bg": "#f5f5f5",
            "gradient_start": "#fb8c00",
            "gradient_end": "#ffd54f",
        },
        "deep": {
            "user_bg": "#e8eaf6",
            "bot_bg": "#f5f5f5",
            "gradient_start": "#3949ab",
            "gradient_end": "#7986cb",
        },
        "forest": {
            "user_bg": "#e8f5e9",
            "bot_bg": "#f1f8e9",
            "gradient_start": "#2e7d32",
            "gradient_end": "#81c784",
        },
        "midnight": {
            "user_bg": "#1c1c1c",
            "bot_bg": "#2a2a2a",
            "gradient_start": "#546e7a",
            "gradient_end": "#90a4ae",
        },
    }
    return themes.get(st.session_state.theme, themes["default"])


def display_chat_history():
    """Display chat history with modern UI"""
    if not st.session_state.history:
        st.info("List your tasks, meetings, time blocks, or energy constraints to get a tailored plan.")
        return

    for msg in st.session_state.history:
        timestamp_display = ""
        if st.session_state.show_timestamps and "timestamp" in msg:
            try:
                dt = datetime.fromisoformat(msg["timestamp"])
                timestamp_display = f" | {dt.strftime('%H:%M')}"
            except ValueError:
                pass

        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"**You{timestamp_display}**")
                st.write(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(f"**Planner{timestamp_display}**")
                st.write(msg["content"])


def render_sidebar():
    """Render sidebar with settings"""
    with st.sidebar:
        st.title("Productivity Settings")

        st.divider()

        st.subheader("Planning Mode")
        planner_modes = ["Balanced Day", "Deep Work Sprint", "Meeting Heavy", "Student Mode"]
        st.session_state.planner_mode = st.selectbox(
            "Mode",
            options=planner_modes,
            index=planner_modes.index(st.session_state.planner_mode),
            help="Guide the planner toward your day type.",
        )

        st.divider()

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

        else:
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

        st.subheader("Appearance")
        theme_options = ["default", "deep", "forest", "midnight"]
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

        st.subheader("Quick Actions")
        if st.button("Sample Prompt", use_container_width=True):
            sample_prompts = {
                "Balanced Day": "Two deep work sessions, a mid day gym break, and evening family time.",
                "Deep Work Sprint": "Need 4 hours of focused writing plus admin tasks before 4 pm.",
                "Meeting Heavy": "Back to back calls 9-2, finish deck, grocery pickup at 6.",
                "Student Mode": "Study calculus, attend lab, part time shift, and review notes at night.",
            }
            prompt = sample_prompts.get(st.session_state.planner_mode, sample_prompts["Balanced Day"])
            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": f"Here is a planning example to try:\n\n{prompt}",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            st.rerun()

        st.divider()

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
            st.subheader("Day Stats")

            total_messages = len(st.session_state.history)
            user_messages = len([m for m in st.session_state.history if m["role"] == "user"])
            bot_messages = total_messages - user_messages

            col1, col2 = st.columns(2)
            col1.metric("Your updates", user_messages)
            col2.metric("Planner notes", bot_messages)

        st.divider()

        st.subheader("Focus Tips")
        st.info(
            "Bundle similar tasks and schedule breaks before you feel drained. "
            "Note energy dips or commute blocks for better pacing."
        )


def export_chat():
    """Export chat history to text file"""
    if not st.session_state.history:
        st.error("No conversation to export")
        return

    export_text = "Daily Productivity Planner Export\n"
    export_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_text += f"Mode: {st.session_state.planner_mode}\n"
    export_text += f"Model: {st.session_state.selected_model}\n"
    export_text += "=" * 50 + "\n\n"

    for msg in st.session_state.history:
        role = "You" if msg["role"] == "user" else "Planner"
        timestamp = msg.get("timestamp", "")
        export_text += f"[{timestamp}] {role}:\n{msg['content']}\n\n"

    st.sidebar.download_button(
        label="Download Session",
        data=export_text,
        file_name=f"daily_planner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
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
                <h1 style='color: #fb8c00; font-size: 3em; margin-bottom: 0;'>Daily Productivity Planner</h1>
                <p style='color: #666; font-size: 1.2em;'>Turn todo lists and meetings into balanced time blocks.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #fb8c00 0%, #ffd54f 100%);
            padding: 20px; border-radius: 15px; color: white; text-align: center; margin-bottom: 30px;'>
                <p style='margin: 0; font-size: 1.1em;'>
                    <strong>Plan smarter every day.</strong><br/>
                    Share tasks, constraints, and energy levels to get a realistic schedule you can iterate on.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        display_chat_history()

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Message",
                placeholder="List meetings, tasks, commute, or habits for today...",
                key=f"user_input_{st.session_state.input_key}",
                label_visibility="collapsed",
                height=140,
                max_chars=2000,
            )

            submitted = st.form_submit_button("Send", use_container_width=True, type="primary")

        if submitted and user_input.strip():
            st.session_state.history.append(
                {
                    "role": "user",
                    "content": user_input.strip(),
                    "timestamp": datetime.now().isoformat(),
                }
            )

            with st.spinner("Planner is arranging your day..."):
                reply = get_planner_response(user_input.strip())

            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": reply,
                    "timestamp": datetime.now().isoformat(),
                }
            )

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
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.2);
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
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.2);
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
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
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
