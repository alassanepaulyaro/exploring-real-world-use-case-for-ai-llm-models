"""
EmpathyBot - Virtual Therapy Chat
A modern AI-powered therapeutic chatbot with multiple model support
"""

import os
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
    page_title="EmpathyBot – Virtual Therapy Chat",
    page_icon="🧠",
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
    "gpt-4-turbo": {"name": "GPT-4 Turbo", "provider": "openai"},
    "gpt-4": {"name": "GPT-4", "provider": "openai"},
    "gpt-3.5-turbo": {"name": "GPT-3.5 Turbo", "provider": "openai"},
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

initialize_session_state()


# ============================
# Model Provider Functions
# ============================

def get_available_ollama_models() -> List[str]:
    """Fetch available Ollama models from the system"""
    if not OLLAMA_AVAILABLE:
        return []

    try:
        models = ollama.list()
        return [model['name'] for model in models.get('models', [])]
    except Exception:
        return list(OLLAMA_MODELS.keys())


def call_ollama(messages: List[Dict], model: str) -> str:
    """Call Ollama API"""
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"Ollama Error: {str(e)}"


def call_openai(messages: List[Dict], model: str) -> str:
    """Call OpenAI API"""
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return "Error: OpenAI API key not found in environment variables."

    try:
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )

        content = response.choices[0].message.content.strip() if response.choices else ""

        if not content:
            return "I'm sorry, I couldn't find the right words just now. Could you try expressing what you're feeling differently?"

        return content

    except Exception as e:
        return f"OpenAI Error: {str(e)}"


# ============================
# Main Chat Function
# ============================

def get_therapy_response(user_input: str) -> str:
    """
    Get therapy response from selected AI model
    """
    system_message = {
        "role": "system",
        "content": (
            "You are a compassionate and empathetic virtual therapist. "
            "You listen carefully, validate the user's emotions, and offer gentle, "
            "supportive reflections and coping suggestions. "
            "Do not diagnose, do not provide medical, legal, or crisis advice. "
            "If the user seems in danger or mentions self-harm, encourage them to "
            "seek immediate help from local emergency services or a trusted person. "
            "Use a calm, warm, non-judgmental tone."
        ),
    }

    # Build message history
    messages = [system_message]

    for msg in st.session_state.history:
        if msg["role"] in ("user", "assistant"):
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

    messages.append({"role": "user", "content": user_input})

    # Route to appropriate provider
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
            "user_bg": "#dcf8c6",
            "bot_bg": "#f1f1f1",
            "gradient_start": "#667eea",
            "gradient_end": "#764ba2",
        },
        "dark": {
            "user_bg": "#005c4b",
            "bot_bg": "#262626",
            "gradient_start": "#4a00e0",
            "gradient_end": "#8e2de2",
        },
        "ocean": {
            "user_bg": "#b3e5fc",
            "bot_bg": "#e1f5fe",
            "gradient_start": "#0097a7",
            "gradient_end": "#00bcd4",
        },
        "sunset": {
            "user_bg": "#ffe0b2",
            "bot_bg": "#fff3e0",
            "gradient_start": "#ff6f00",
            "gradient_end": "#ff9100",
        },
    }
    return themes.get(st.session_state.theme, themes["default"])


def display_chat_history():
    """Display chat history with modern UI"""
    if not st.session_state.history:
        st.info("🌟 Start by sharing what's on your mind. You can talk about your day, emotions, or anything that feels heavy.")
        return

    for msg in st.session_state.history:
        timestamp_display = ""
        if st.session_state.show_timestamps and "timestamp" in msg:
            try:
                dt = datetime.fromisoformat(msg["timestamp"])
                timestamp_display = f" • {dt.strftime('%H:%M')}"
            except:
                pass

        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"**You{timestamp_display}**")
                st.write(msg['content'])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(f"**EmpathyBot{timestamp_display}**")
                st.write(msg['content'])


def render_sidebar():
    """Render sidebar with settings"""
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/100/000000/brain.png", width=80)
        st.title("⚙️ Settings")

        st.divider()

        # Model Provider Selection
        st.subheader("🤖 Model Selection")

        provider_options = {
            "Ollama (Cloud)": "ollama_cloud",
            "Ollama (Local)": "ollama",
            "OpenAI": "openai",
        }

        selected_provider_name = st.selectbox(
            "Provider",
            options=list(provider_options.keys()),
            index=0,
            help="Choose your AI model provider"
        )

        selected_provider = provider_options[selected_provider_name]

        # Model selection based on provider
        if selected_provider == "openai":
            available_models = {k: v["name"] for k, v in OPENAI_MODELS.items()}
            model_key = st.selectbox(
                "Model",
                options=list(available_models.keys()),
                format_func=lambda x: available_models[x],
                help="Select OpenAI model"
            )
            st.session_state.model_provider = "openai"
            st.session_state.selected_model = model_key

            # Show API key status
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                st.success("✅ API Key loaded from .env")
            else:
                st.error("❌ No API key found in .env")

        elif selected_provider == "ollama":
            local_models = {k: v["name"] for k, v in OLLAMA_MODELS.items() if v["provider"] == "ollama"}
            if local_models:
                model_key = st.selectbox(
                    "Model",
                    options=list(local_models.keys()),
                    format_func=lambda x: local_models[x],
                    help="Select local Ollama model"
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
                index=0,  # Default to gpt-oss:120b-cloud
                help="Select Ollama cloud model"
            )
            st.session_state.model_provider = "ollama_cloud"
            st.session_state.selected_model = model_key

        # Display current selection
        st.info(f"**Current Model:**\n{st.session_state.selected_model}")

        st.divider()

        # UI Customization
        st.subheader("🎨 Appearance")

        theme_options = ["default", "dark", "ocean", "sunset"]
        st.session_state.theme = st.selectbox(
            "Theme",
            options=theme_options,
            index=theme_options.index(st.session_state.theme),
            format_func=lambda x: x.title()
        )

        st.session_state.show_timestamps = st.checkbox(
            "Show timestamps",
            value=st.session_state.show_timestamps
        )

        st.divider()

        # Session Controls
        st.subheader("💬 Session Controls")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 New Chat", use_container_width=True):
                st.session_state.history = []
                st.session_state.input_key += 1
                st.rerun()

        with col2:
            if st.button("📥 Export", use_container_width=True):
                if st.session_state.history:
                    export_chat()

        # Statistics
        if st.session_state.history:
            st.divider()
            st.subheader("📊 Statistics")

            total_messages = len(st.session_state.history)
            user_messages = len([m for m in st.session_state.history if m["role"] == "user"])
            bot_messages = total_messages - user_messages

            col1, col2 = st.columns(2)
            col1.metric("Your messages", user_messages)
            col2.metric("Bot messages", bot_messages)

        st.divider()

        # Important Notice
        st.subheader("⚠️ Important Notice")
        st.warning(
            "This chatbot provides emotional support only. "
            "It does not give medical advice. "
            "In case of emergency, contact local services immediately."
        )


def export_chat():
    """Export chat history to text file"""
    if not st.session_state.history:
        st.error("No conversation to export")
        return

    export_text = f"EmpathyBot Conversation Export\n"
    export_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_text += f"Model: {st.session_state.selected_model}\n"
    export_text += "=" * 50 + "\n\n"

    for msg in st.session_state.history:
        role = "You" if msg["role"] == "user" else "EmpathyBot"
        timestamp = msg.get("timestamp", "")
        export_text += f"[{timestamp}] {role}:\n{msg['content']}\n\n"

    st.sidebar.download_button(
        label="💾 Download Chat",
        data=export_text,
        file_name=f"empathybot_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True
    )


# ============================
# Main Application
# ============================

def main():
    """Main application logic"""

    # Render sidebar
    render_sidebar()

    # Main content area
    _, col2, _ = st.columns([1, 6, 1])

    with col2:
        st.markdown(
            """
            <div style='text-align: center; padding: 20px;'>
                <h1 style='color: #667eea; font-size: 3em; margin-bottom: 0;'>🧠 EmpathyBot</h1>
                <p style='color: #888; font-size: 1.2em;'>Your Virtual Therapy Companion</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px; border-radius: 15px; color: white; text-align: center; margin-bottom: 30px;'>
                <p style='margin: 0; font-size: 1.1em;'>
                    This space is designed for <strong>emotional support and gentle reflection</strong>.<br/>
                    It is <strong>not a substitute for professional mental health care</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display chat history
        display_chat_history()

        # Chat input form
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Message",
                placeholder="What would you like to share today?",
                key=f"user_input_{st.session_state.input_key}",
                label_visibility="collapsed",
                height=100,
                max_chars=2000
            )

            submitted = st.form_submit_button("📤 Send", use_container_width=True, type="primary")

        # Handle message submission
        if submitted and user_input.strip():
            # Add user message to history
            st.session_state.history.append({
                "role": "user",
                "content": user_input.strip(),
                "timestamp": datetime.now().isoformat(),
            })

            # Get bot response
            with st.spinner("🧠 EmpathyBot is reflecting on your words..."):
                reply = get_therapy_response(user_input.strip())

            # Add bot response to history
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
        /* Main container */
        .main .block-container {{
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }}

        /* Input styling */
        .stTextInput > div > div > input {{
            border-radius: 25px;
            border: 2px solid #e0e0e0;
            padding: 12px 20px;
            font-size: 1em;
            transition: all 0.3s ease;
        }}

        .stTextInput > div > div > input:focus {{
            border-color: {colors["gradient_start"]};
            box-shadow: 0 0 15px rgba(102, 126, 234, 0.3);
        }}

        /* Text area styling */
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
            box-shadow: 0 0 15px rgba(102, 126, 234, 0.3);
        }}

        /* Button styling */
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
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }}

        /* Selectbox styling */
        .stSelectbox > div > div {{
            border-radius: 10px;
            border: 2px solid #e0e0e0;
        }}

        /* Sidebar styling */
        .css-1d391kg {{
            padding-top: 2rem;
        }}

        /* Scrollbar */
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

        /* Form styling */
        .stForm {{
            border: none;
            padding: 20px 0;
        }}

        /* Metrics */
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
