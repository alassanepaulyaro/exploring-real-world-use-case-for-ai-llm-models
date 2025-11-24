# Shopping List Creator
"""
Smart shopping list builder for any shopping needs - groceries, household items,
office supplies, travel essentials, and more. Categorized, organized, and budget-friendly.
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
    page_title="Shopping List Creator",
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

    if "list_mode" not in st.session_state:
        st.session_state.list_mode = "Groceries & Food"


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
            return "I could not build your list yet - include meals, servings, or pantry notes."

        return clean_html_tags(content)

    except Exception as e:
        return f"OpenAI Error: {str(e)}"


# ============================
# Main Chat Function
# ============================


def get_shopping_list(user_input: str) -> str:
    """Route conversation through selected AI model"""
    list_modes = {
        "Groceries & Food": (
            "You are a smart grocery shopping assistant. Create organized shopping lists for food and groceries. "
            "Categorize items by store section (produce, dairy, meat, pantry, frozen, bakery). Include quantities, "
            "suggest meal prep tips, and note storage recommendations. Consider dietary needs, household size, and budget. "
            "Format your responses using plain text with markdown. Never use HTML tags."
        ),
        "Household Essentials": (
            "You are a household supplies organizer. Create shopping lists for cleaning supplies, toiletries, paper goods, "
            "laundry items, and home maintenance products. Organize by category (bathroom, kitchen, laundry room, cleaning). "
            "Suggest bulk buying opportunities, eco-friendly alternatives, and budget-friendly options. Note which items "
            "can be bought in multi-packs or generic brands. "
            "Format your responses using plain text with markdown. Never use HTML tags."
        ),
        "Travel & Vacation": (
            "You are a travel packing expert. Create comprehensive packing and shopping lists for trips. Include clothing, "
            "toiletries, electronics, documents, medications, and destination-specific items. Organize by category and priority. "
            "Consider trip duration, weather, activities planned, and airline restrictions. Suggest travel-sized options and "
            "items that can be purchased at the destination to save luggage space. "
            "Format your responses using plain text with markdown. Never use HTML tags."
        ),
        "Office & School Supplies": (
            "You are an office and school supplies organizer. Create shopping lists for stationery, electronics, organizational "
            "tools, and educational materials. Categorize by type (writing instruments, paper products, tech accessories, "
            "storage solutions). Suggest bulk buying for frequently used items, budget-friendly alternatives, and quality "
            "brands for important tools. Consider back-to-school seasons or office setup needs. "
            "Format your responses using plain text with markdown. Never use HTML tags."
        ),
        "Party & Events": (
            "You are an event planning assistant. Create shopping lists for parties, celebrations, and special events. "
            "Include decorations, tableware, food & beverages, party favors, and entertainment supplies. Organize by category "
            "and priority. Consider guest count, theme, budget, and venue requirements. Suggest DIY options, bulk buying "
            "opportunities, and vendor recommendations for larger items. "
            "Format your responses using plain text with markdown. Never use HTML tags."
        ),
        "Home Improvement": (
            "You are a home improvement shopping guide. Create lists for DIY projects, home repairs, and renovations. "
            "Include tools, materials, hardware, safety equipment, and supplies. Organize by project phase or room. "
            "Provide quantity estimates, alternative materials, and budget considerations. Note which items might require "
            "professional assistance or special permits. Suggest tool rentals vs. purchases for one-time projects. "
            "Format your responses using plain text with markdown. Never use HTML tags."
        ),
    }

    system_message = {
        "role": "system",
        "content": list_modes.get(st.session_state.list_mode, list_modes["Groceries & Food"]),
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
            "user_bg": "#e8f5e9",
            "bot_bg": "#f5f5f5",
            "gradient_start": "#43a047",
            "gradient_end": "#81c784",
        },
        "market": {
            "user_bg": "#fff3e0",
            "bot_bg": "#fff8e1",
            "gradient_start": "#fb8c00",
            "gradient_end": "#ffd54f",
        },
        "midnight": {
            "user_bg": "#1c1c1c",
            "bot_bg": "#2a2a2a",
            "gradient_start": "#546e7a",
            "gradient_end": "#90a4ae",
        },
        "aqua": {
            "user_bg": "#e0f7fa",
            "bot_bg": "#f1fbff",
            "gradient_start": "#00838f",
            "gradient_end": "#4dd0e1",
        },
    }
    return themes.get(st.session_state.theme, themes["default"])


def display_chat_history():
    """Display chat history"""
    if not st.session_state.history:
        st.info("Tell me what you need to shop for - groceries, household items, travel essentials, or any shopping list - and I'll organize it for you!")
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
                st.markdown(f"**List Bot{timestamp_display}**")
                st.write(msg["content"])


def render_sidebar():
    """Render sidebar with settings"""
    with st.sidebar:
        st.title("Shopping Settings")

        st.divider()

        st.subheader("Shopping Category")
        list_modes = ["Groceries & Food", "Household Essentials", "Travel & Vacation", "Office & School Supplies", "Party & Events", "Home Improvement"]
        st.session_state.list_mode = st.selectbox(
            "Category",
            options=list_modes,
            index=list_modes.index(st.session_state.list_mode) if st.session_state.list_mode in list_modes else 0,
            help="Choose the type of shopping list you need.",
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
        theme_options = ["default", "market", "midnight", "aqua"]
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
                "Groceries & Food": "Weekly groceries for a family of 4: breakfast, lunches, dinners, and healthy snacks. Budget $150.",
                "Household Essentials": "Restocking bathroom and kitchen cleaning supplies, plus paper towels and toiletries for the month.",
                "Travel & Vacation": "2-week beach vacation to Hawaii for 2 adults. Need packing list and items to buy before trip.",
                "Office & School Supplies": "Back-to-school supplies for 2 kids (grades 3 and 7) plus home office essentials.",
                "Party & Events": "Birthday party for 30 people with outdoor BBQ theme. Need decorations, food, drinks, and supplies.",
                "Home Improvement": "Painting bedroom walls and installing new curtain rods. List all tools and materials needed.",
            }
            prompt = sample_prompts.get(st.session_state.list_mode, sample_prompts["Groceries & Food"])
            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": f"Try asking:\n\n{prompt}",
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
            st.subheader("List Stats")

            total_messages = len(st.session_state.history)
            user_messages = len([m for m in st.session_state.history if m["role"] == "user"])
            bot_messages = total_messages - user_messages

            col1, col2 = st.columns(2)
            col1.metric("Your plans", user_messages)
            col2.metric("Lists generated", bot_messages)

        st.divider()

        st.subheader("Shopping Tips")
        st.info(
            "Be specific about quantities, budget, timeline, or special requirements. "
            "Mention items you already have or store preferences for better recommendations."
        )


def export_chat():
    """Export chat history to text file"""
    if not st.session_state.history:
        st.error("No conversation to export")
        return

    export_text = "Shopping List Session Export\n"
    export_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_text += f"Mode: {st.session_state.list_mode}\n"
    export_text += f"Model: {st.session_state.selected_model}\n"
    export_text += "=" * 50 + "\n\n"

    for msg in st.session_state.history:
        role = "You" if msg["role"] == "user" else "List Bot"
        timestamp = msg.get("timestamp", "")
        export_text += f"[{timestamp}] {role}:\n{msg['content']}\n\n"

    st.sidebar.download_button(
        label="Download Session",
        data=export_text,
        file_name=f"shopping_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
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
                <h1 style='color: #43a047; font-size: 3em; margin-bottom: 0;'>Smart Shopping List Creator</h1>
                <p style='color: #666; font-size: 1.2em;'>Organized lists for any shopping need - from groceries to travel essentials</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #43a047 0%, #81c784 100%);
            padding: 20px; border-radius: 15px; color: white; text-align: center; margin-bottom: 30px;'>
                <p style='margin: 0; font-size: 1.1em;'>
                    <strong>Shop smarter, save time.</strong><br/>
                    Create organized, categorized shopping lists for groceries, household items, travel, events, and more.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        display_chat_history()

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Message",
                placeholder="Describe what you need to shop for: groceries, household items, travel packing, party supplies, etc...",
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

            with st.spinner("Shopping Bot is organizing your list..."):
                reply = get_shopping_list(user_input.strip())

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
