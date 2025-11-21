# Twitter Assistant
"""
Condenses multi-tweet threads into concise TLDRs, bullet lists, action items, or narratives.
Also generates catchy, engaging tweets based on your topic, audience, and style.
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
    page_title="Twitter Assistant",
    page_icon="🐦",
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

    if "summary_mode" not in st.session_state:
        st.session_state.summary_mode = "TLDR"

    if "twitter_tool" not in st.session_state:
        st.session_state.twitter_tool = "Twitter Summarizer"

    if "tweet_tone" not in st.session_state:
        st.session_state.tweet_tone = "Professional"


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
            return "I could not summarize that yet - mind pasting the thread again?"

        return clean_html_tags(content)

    except Exception as e:
        return f"OpenAI Error: {str(e)}"


# ============================
# Main Chat Function
# ============================


def get_summary_response(user_input: str) -> str:
    """Route conversation through the selected AI model"""

    # Determine the system prompt based on selected Twitter tool
    if st.session_state.twitter_tool == "Tweet Generator":
        tweet_tones = {
            "Professional": (
                "You are a professional Twitter copywriter who generates polished, business-appropriate tweets. "
                "Your specialty is crafting content that fits the 280-character Twitter limit while maintaining "
                "a professional tone suitable for corporate, B2B, or thought leadership contexts. Focus on clear "
                "value propositions, industry insights, and credible messaging. Generate 2-4 tweet options with "
                "different angles while keeping the professional tone consistent. Use relevant hashtags strategically "
                "and avoid casual slang or excessive emojis. Prioritize clarity, authority, and professionalism. "
                "Format your responses using plain text with markdown. Never use HTML tags like <br>, <b>, <i>, etc. "
                "Use markdown syntax instead (line breaks, **bold**, *italic*)."
            ),
            "Friendly": (
                "You are a warm and approachable Twitter copywriter who generates friendly, conversational tweets. "
                "Your specialty is crafting content that fits the 280-character Twitter limit while being welcoming, "
                "relatable, and personable. Create tweets that feel like they're coming from a friend—use casual "
                "language, light humor, and emojis when appropriate. Generate 2-4 tweet options with different angles "
                "that encourage friendly engagement and community building. Focus on authenticity, positivity, and "
                "making connections with the audience. "
                "Format your responses using plain text with markdown. Never use HTML tags like <br>, <b>, <i>, etc. "
                "Use markdown syntax instead (line breaks, **bold**, *italic*)."
            ),
            "Concise": (
                "You are a master of brevity—a Twitter copywriter who generates ultra-concise, impactful tweets. "
                "Your specialty is crafting sharp, direct content that fits well within the 280-character limit, "
                "often much shorter. Every word counts. Cut the fluff, get straight to the point, and deliver "
                "maximum impact with minimum words. Generate 2-4 tweet options that are punchy, clear, and memorable. "
                "Use short sentences, active voice, and powerful verbs. No unnecessary words or elaboration. "
                "Format your responses using plain text with markdown. Never use HTML tags like <br>, <b>, <i>, etc. "
                "Use markdown syntax instead (line breaks, **bold**, *italic*)."
            ),
            "Polite": (
                "You are a courteous and respectful Twitter copywriter who generates polite, considerate tweets. "
                "Your specialty is crafting content that fits the 280-character Twitter limit while maintaining "
                "respectful, tactful, and gracious communication. Use courteous language, acknowledge different "
                "perspectives, and express ideas with kindness and diplomacy. Generate 2-4 tweet options that "
                "balance assertiveness with politeness. Perfect for sensitive topics, customer service, or "
                "maintaining positive brand reputation. Prioritize respect, empathy, and constructive communication. "
                "Format your responses using plain text with markdown. Never use HTML tags like <br>, <b>, <i>, etc. "
                "Use markdown syntax instead (line breaks, **bold**, *italic*)."
            ),
        }
        system_prompt = tweet_tones.get(st.session_state.tweet_tone, tweet_tones["Professional"])
    else:  # Twitter Summarizer
        summary_modes = {
            "TLDR": (
                "You are a social media analyst who produces concise TLDR summaries of long Twitter threads. "
                "Capture the core argument, supporting evidence, and conclusion in 3 to 4 sentences. Use markdown."
            ),
            "Key Points": (
                "You summarize threads into bullet lists with clear headings. "
                "Highlight main claims, data points, and quotes. Keep bullets punchy. Use markdown."
            ),
            "Storyline": (
                "You retell the thread as a narrative with beginning, conflict, and resolution. "
                "Reference the author's tone and include context readers need. Use markdown."
            ),
            "Action Items": (
                "You extract practical steps or lessons from the thread. "
                "List prioritized actions, resources, or metrics people should watch. Use markdown."
            ),
        }
        system_prompt = summary_modes.get(st.session_state.summary_mode, summary_modes["TLDR"])

    system_message = {
        "role": "system",
        "content": system_prompt,
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
            "user_bg": "#e0f2f1",
            "bot_bg": "#f5f5f5",
            "gradient_start": "#00897b",
            "gradient_end": "#4db6ac",
        },
        "slate": {
            "user_bg": "#eceff1",
            "bot_bg": "#fafafa",
            "gradient_start": "#455a64",
            "gradient_end": "#78909c",
        },
        "night": {
            "user_bg": "#1c2331",
            "bot_bg": "#0f1724",
            "gradient_start": "#3949ab",
            "gradient_end": "#5c6bc0",
        },
        "citrus": {
            "user_bg": "#fffde7",
            "bot_bg": "#fff8e1",
            "gradient_start": "#f9a825",
            "gradient_end": "#ffca28",
        },
        "generator": {
            "user_bg": "#e3f2fd",
            "bot_bg": "#fafafa",
            "gradient_start": "#1da1f2",
            "gradient_end": "#42a5f5",
        },
    }
    return themes.get(st.session_state.theme, themes["default"])


def display_chat_history():
    """Display chat history with modern UI"""
    if not st.session_state.history:
        st.info("Summarize Twitter threads or generate engaging tweets! Choose your tool from the sidebar.")
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
                st.markdown(f"**Twitter Assistant{timestamp_display}**")
                st.write(msg["content"])


def render_sidebar():
    """Render sidebar with settings"""
    with st.sidebar:
        st.title("Twitter Tools")

        st.divider()

        # Twitter Tool Selection
        st.subheader("Twitter Tool")
        twitter_tools = ["Twitter Summarizer", "Tweet Generator"]
        st.session_state.twitter_tool = st.selectbox(
            "Tool",
            options=twitter_tools,
            index=twitter_tools.index(st.session_state.twitter_tool),
            help="Choose between summarizing threads or generating new tweets.",
        )

        st.divider()

        # Show appropriate mode selection based on Twitter Tool
        if st.session_state.twitter_tool == "Twitter Summarizer":
            st.subheader("Summary Mode")
            summary_modes = ["TLDR", "Key Points", "Storyline", "Action Items"]
            st.session_state.summary_mode = st.selectbox(
                "Mode",
                options=summary_modes,
                index=summary_modes.index(st.session_state.summary_mode) if st.session_state.summary_mode in summary_modes else 0,
                help="Choose the type of summary you want.",
            )

            st.divider()

        elif st.session_state.twitter_tool == "Tweet Generator":
            st.subheader("Tweet Tone")
            tweet_tones = ["Professional", "Friendly", "Concise", "Polite"]
            st.session_state.tweet_tone = st.selectbox(
                "Tone",
                options=tweet_tones,
                index=tweet_tones.index(st.session_state.tweet_tone) if st.session_state.tweet_tone in tweet_tones else 0,
                help="Choose the tone style for generated tweets.",
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
            st.subheader("Session Stats")

            total_messages = len(st.session_state.history)
            user_messages = len([m for m in st.session_state.history if m["role"] == "user"])
            bot_messages = total_messages - user_messages

            col1, col2 = st.columns(2)
            col1.metric("Your requests", user_messages)
            col2.metric("AI responses", bot_messages)


def export_chat():
    """Export chat history to text file"""
    if not st.session_state.history:
        st.error("No conversation to export")
        return

    export_text = "Twitter Assistant Export\n"
    export_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_text += f"Tool: {st.session_state.twitter_tool}\n"
    if st.session_state.twitter_tool == "Twitter Summarizer":
        export_text += f"Summary Mode: {st.session_state.summary_mode}\n"
    elif st.session_state.twitter_tool == "Tweet Generator":
        export_text += f"Tweet Tone: {st.session_state.tweet_tone}\n"
    export_text += f"Model: {st.session_state.selected_model}\n"
    export_text += "=" * 50 + "\n\n"

    for msg in st.session_state.history:
        role = "You" if msg["role"] == "user" else "Twitter Assistant"
        timestamp = msg.get("timestamp", "")
        export_text += f"[{timestamp}] {role}:\n{msg['content']}\n\n"

    st.sidebar.download_button(
        label="Download Session",
        data=export_text,
        file_name=f"twitter_assistant_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
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
                <h1 style='color: #00897b; font-size: 3em; margin-bottom: 0;'>🐦 Twitter Assistant</h1>
                <p style='color: #666; font-size: 1.2em;'>Summarize threads and generate engaging tweets with AI</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #00897b 0%, #4db6ac 100%);
            padding: 20px; border-radius: 15px; color: white; text-align: center; margin-bottom: 30px;'>
                <p style='margin: 0; font-size: 1.1em;'>
                    <strong>Master Twitter content.</strong><br/>
                    Condense threads into insights or craft catchy tweets that engage your audience.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        display_chat_history()

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Message",
                placeholder="Paste a Twitter thread to summarize, or describe a topic to generate tweets...",
                key=f"user_input_{st.session_state.input_key}",
                label_visibility="collapsed",
                height=200,
                max_chars=4000,
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

            with st.spinner("Twitter Assistant is working..."):
                reply = get_summary_response(user_input.strip())

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

