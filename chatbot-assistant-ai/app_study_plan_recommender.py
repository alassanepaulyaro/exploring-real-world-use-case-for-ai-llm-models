# Study Plan Recommender
"""
Builds structured study schedules tailored to goals, timelines, and learning styles.
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
    page_title="Study Plan Recommender",
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

    if "plan_mode" not in st.session_state:
        st.session_state.plan_mode = "Exam Prep"


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
            return "I could not craft a plan yet - include subject, timeline, or availability."

        return clean_html_tags(content)

    except Exception as e:
        return f"OpenAI Error: {str(e)}"


# ============================
# Main Chat Function
# ============================


def get_plan_response(user_input: str) -> str:
    """Route conversation through the selected AI model"""
    plan_modes = {
        "Exam Prep": (
            "You are an academic coach creating exam study plans. "
            "Break content into weekly milestones, include review cycles, and add practice tips. Use markdown."
        ),
        "Skill Sprint": (
            "You design intensive short term learning sprints (2-6 weeks). "
            "Mix theory, projects, and reflection checkpoints. Use markdown."
        ),
        "Slow and Steady": (
            "You build gentle routines for busy learners balancing work and study. "
            "Suggest low friction habits and weekend deep dives. Use markdown."
        ),
        "Catch Up": (
            "You help students recover when behind on coursework. "
            "Prioritize core topics, triage optional material, and include sanity breaks. Use markdown."
        ),
    }

    system_message = {
        "role": "system",
        "content": plan_modes.get(st.session_state.plan_mode, plan_modes["Exam Prep"]),
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
            "user_bg": "#e3f2fd",
            "bot_bg": "#f5f5f5",
            "gradient_start": "#1976d2",
            "gradient_end": "#64b5f6",
        },
        "sunrise": {
            "user_bg": "#fff8e1",
            "bot_bg": "#fffde7",
            "gradient_start": "#ffb300",
            "gradient_end": "#ffca28",
        },
        "twilight": {
            "user_bg": "#ede7f6",
            "bot_bg": "#f3e5f5",
            "gradient_start": "#7e57c2",
            "gradient_end": "#ba68c8",
        },
        "forest": {
            "user_bg": "#e8f5e9",
            "bot_bg": "#f1f8e9",
            "gradient_start": "#2e7d32",
            "gradient_end": "#66bb6a",
        },
    }
    return themes.get(st.session_state.theme, themes["default"])


def display_chat_history():
    """Display chat history with modern UI"""
    if not st.session_state.history:
        st.info("Mention your subject, timeline, available hours, and learning preferences to get a custom plan.")
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
                st.markdown(f"**Coach{timestamp_display}**")
                st.write(msg["content"])


def render_sidebar():
    """Render sidebar with settings"""
    with st.sidebar:
        st.title("Planning Settings")

        st.divider()

        st.subheader("Plan Mode")
        plan_modes = ["Exam Prep", "Skill Sprint", "Slow and Steady", "Catch Up"]
        st.session_state.plan_mode = st.selectbox(
            "Mode",
            options=plan_modes,
            index=plan_modes.index(st.session_state.plan_mode),
            help="Choose the planning style that fits your goal.",
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
        theme_options = ["default", "sunrise", "twilight", "forest"]
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
                "Exam Prep": "Plan for AP Biology in 6 weeks with 2 hours free each weekday and full Sundays.",
                "Skill Sprint": "Learn Figma basics in 3 weeks with nightly 90 minute sessions.",
                "Slow and Steady": "Master conversational Spanish over 4 months while working full time.",
                "Catch Up": "I am 4 lectures behind in statistics, need to be ready for a quiz next week.",
            }
            prompt = sample_prompts.get(st.session_state.plan_mode, sample_prompts["Exam Prep"])
            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": f"Prompt idea:\n\n{prompt}",
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
            st.subheader("Study Stats")

            total_messages = len(st.session_state.history)
            user_messages = len([m for m in st.session_state.history if m["role"] == "user"])
            bot_messages = total_messages - user_messages

            col1, col2 = st.columns(2)
            col1.metric("Your inputs", user_messages)
            col2.metric("Coach replies", bot_messages)

        st.divider()

        st.subheader("Learning Tips")
        st.info(
            "Include learning style (visual, practice heavy), available tools, and assessment dates. "
            "Add reminders for rest days to avoid burnout."
        )


def export_chat():
    """Export chat history to text file"""
    if not st.session_state.history:
        st.error("No conversation to export")
        return

    export_text = "Study Plan Export\n"
    export_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_text += f"Mode: {st.session_state.plan_mode}\n"
    export_text += f"Model: {st.session_state.selected_model}\n"
    export_text += "=" * 50 + "\n\n"

    for msg in st.session_state.history:
        role = "You" if msg["role"] == "user" else "Coach"
        timestamp = msg.get("timestamp", "")
        export_text += f"[{timestamp}] {role}:\n{msg['content']}\n\n"

    st.sidebar.download_button(
        label="Download Session",
        data=export_text,
        file_name=f"study_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
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
                <h1 style='color: #1976d2; font-size: 3em; margin-bottom: 0;'>Study Plan Recommender</h1>
                <p style='color: #666; font-size: 1.2em;'>Get a structured roadmap for exams, skills, or catch-up missions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #1976d2 0%, #64b5f6 100%);
            padding: 20px; border-radius: 15px; color: white; text-align: center; margin-bottom: 30px;'>
                <p style='margin: 0; font-size: 1.1em;'>
                    <strong>Plan with intention.</strong><br/>
                    Outline your goals, time budget, and tricky topics to receive a realistic schedule plus checkpoints.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        display_chat_history()

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Message",
                placeholder="Example: Prepare for PMP exam in 10 weeks with evenings free...",
                key=f"user_input_{st.session_state.input_key}",
                label_visibility="collapsed",
                height=140,
                max_chars=2500,
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

            with st.spinner("Coach is drafting your study blueprint..."):
                reply = get_plan_response(user_input.strip())

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
