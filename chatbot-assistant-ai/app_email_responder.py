# AI-Powered Email Responder
import streamlit as st
import requests
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# Ollama API URL (local)
OLLAMA_URL = "http://localhost:11434/api/generate"

# Language code to name mapping
LANGUAGE_NAMES = {
    'af': 'Afrikaans', 'ar': 'Arabic', 'bg': 'Bulgarian', 'bn': 'Bengali',
    'ca': 'Catalan', 'cs': 'Czech', 'cy': 'Welsh', 'da': 'Danish',
    'de': 'German', 'el': 'Greek', 'en': 'English', 'es': 'Spanish',
    'et': 'Estonian', 'fa': 'Persian', 'fi': 'Finnish', 'fr': 'French',
    'gu': 'Gujarati', 'he': 'Hebrew', 'hi': 'Hindi', 'hr': 'Croatian',
    'hu': 'Hungarian', 'id': 'Indonesian', 'it': 'Italian', 'ja': 'Japanese',
    'kn': 'Kannada', 'ko': 'Korean', 'lt': 'Lithuanian', 'lv': 'Latvian',
    'mk': 'Macedonian', 'ml': 'Malayalam', 'mr': 'Marathi', 'ne': 'Nepali',
    'nl': 'Dutch', 'no': 'Norwegian', 'pa': 'Punjabi', 'pl': 'Polish',
    'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 'sk': 'Slovak',
    'sl': 'Slovenian', 'so': 'Somali', 'sq': 'Albanian', 'sv': 'Swedish',
    'sw': 'Swahili', 'ta': 'Tamil', 'te': 'Telugu', 'th': 'Thai',
    'tl': 'Tagalog', 'tr': 'Turkish', 'uk': 'Ukrainian', 'ur': 'Urdu',
    'vi': 'Vietnamese', 'zh-cn': 'Chinese (Simplified)', 'zh-tw': 'Chinese (Traditional)'
}

def detect_language(text):
    """
    Detects the language code and returns code + human name.
    """
    try:
        lang_code = detect(text)
        lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)
        return lang_code, lang_name
    except LangDetectException:
        return 'en', 'English'  # Fallback to English

def generate_email_response(email_content, tone="Formal", language="English"):
    """
    Uses Mistral-Nemo to generate an email reply in the selected tone and detected language.
    """
    prompt = (
        f"Generate an {tone.lower()} email as a response from the customer support team to the customer "
        f"for the following email:\n\n{email_content}\n\n"
        f"Write the response in {language}. Ensure the response is polite, clear, and professional."
    )

    payload = {
        "model": "mistral-nemo:latest",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "No response generated.")
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Request failed: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="AI-Powered Email Responder", layout="centered")

st.title("AI-Powered Email Responder")
st.write("Paste an email and let the AI generate a professional response.")

with st.form("email_form"):
    email_content = st.text_area("Paste the email content here", height=200)
    tone = st.radio(
        "Select Tone",
        options=["Formal", "Casual", "Friendly", "Professional", "Assertive", "Humorous"],
        index=3
    )
    submitted = st.form_submit_button("Generate Response")

    if email_content.strip():
        lang_code, lang_name = detect_language(email_content)
        st.info(f"🔍 Detected language: **{lang_name}** ({lang_code})")
    else:
        lang_name = "English"

if submitted:
    if email_content.strip() == "":
        st.warning("Please paste the email content first.")
    else:
        with st.spinner(f"Generating response in {lang_name}..."):
            response = generate_email_response(email_content, tone, lang_name)
        st.text_area("AI-Generated Email Response", value=response, height=200)
