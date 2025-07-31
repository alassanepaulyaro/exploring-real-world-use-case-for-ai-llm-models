# Customer Support Chatbot with Multilanguage Support

# Customer Support Chatbot with Multilanguage Support

import streamlit as st
import requests
import json
import time
from datetime import datetime

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"
MAX_HISTORY = 50

# Language configurations
LANGUAGES = {
    "🇺🇸 English": {
        "code": "en",
        "name": "English",
        "title": "🤖 Customer Support Chatbot",
        "subtitle": "💬 Ask your question about our products or services. Our AI assistant will respond professionally and remember our conversation.",
        "input_placeholder": "E.g.: How can I track my order?",
        "send_button": "📤 Send",
        "new_conversation": "🔄 New Conversation",
        "export_chat": "📄 Export Chat",
        "download_chat": "💾 Download Chat",
        "chat_controls": "💬 Chat Controls",
        "chat_stats": "📊 Chat Stats",
        "total_messages": "Total Messages",
        "your_questions": "Your Questions",
        "information": "ℹ️ Information",
        "info_text": "This chatbot uses Llama 3.1 8B model running locally via Ollama.",
        "thinking": "🤖 Chatbot is thinking...",
        "start_conversation": "👋 Start a conversation by asking a question!",
        "helpful_response": "Helpful response",
        "not_helpful": "Not helpful",
        "feedback_positive": "Thanks for your feedback!",
        "feedback_negative": "Thanks for your feedback. We'll improve!",
        "export_header": "Customer Support Chat Export"
    },
    "🇪🇸 Español": {
        "code": "es",
        "name": "Spanish",
        "title": "🤖 Chatbot de Soporte al Cliente",
        "subtitle": "💬 Haz tu pregunta sobre nuestros productos o servicios. Nuestro asistente IA responderá profesionalmente y recordará nuestra conversación.",
        "input_placeholder": "Ej.: ¿Cómo puedo rastrear mi pedido?",
        "send_button": "📤 Enviar",
        "new_conversation": "🔄 Nueva Conversación",
        "export_chat": "📄 Exportar Chat",
        "download_chat": "💾 Descargar Chat",
        "chat_controls": "💬 Controles de Chat",
        "chat_stats": "📊 Estadísticas del Chat",
        "total_messages": "Mensajes Totales",
        "your_questions": "Tus Preguntas",
        "information": "ℹ️ Información",
        "info_text": "Este chatbot usa el modelo Llama 3.1 8B ejecutándose localmente via Ollama.",
        "thinking": "🤖 El chatbot está pensando...",
        "start_conversation": "👋 ¡Inicia una conversación haciendo una pregunta!",
        "helpful_response": "Respuesta útil",
        "not_helpful": "No útil",
        "feedback_positive": "¡Gracias por tu feedback!",
        "feedback_negative": "Gracias por tu feedback. ¡Mejoraremos!",
        "export_header": "Exportación de Chat de Soporte al Cliente"
    },
    "🇫🇷 Français": {
        "code": "fr",
        "name": "French",
        "title": "🤖 Chatbot de Support Client",
        "subtitle": "💬 Posez votre question sur nos produits ou services. Notre assistant IA répondra professionnellement et se souviendra de notre conversation.",
        "input_placeholder": "Ex.: Comment puis-je suivre ma commande?",
        "send_button": "📤 Envoyer",
        "new_conversation": "🔄 Nouvelle Conversation",
        "export_chat": "📄 Exporter Chat",
        "download_chat": "💾 Télécharger Chat",
        "chat_controls": "💬 Contrôles de Chat",
        "chat_stats": "📊 Statistiques du Chat",
        "total_messages": "Messages Totaux",
        "your_questions": "Vos Questions",
        "information": "ℹ️ Information",
        "info_text": "Ce chatbot utilise le modèle Llama 3.1 8B fonctionnant localement via Ollama.",
        "thinking": "🤖 Le chatbot réfléchit...",
        "start_conversation": "👋 Commencez une conversation en posant une question!",
        "helpful_response": "Réponse utile",
        "not_helpful": "Pas utile",
        "feedback_positive": "Merci pour votre feedback!",
        "feedback_negative": "Merci pour votre feedback. Nous améliorerons!",
        "export_header": "Export de Chat de Support Client"
    },
    "🇩🇪 Deutsch": {
        "code": "de",
        "name": "German",
        "title": "🤖 Kundensupport Chatbot",
        "subtitle": "💬 Stellen Sie Ihre Frage zu unseren Produkten oder Dienstleistungen. Unser KI-Assistent wird professionell antworten und sich an unser Gespräch erinnern.",
        "input_placeholder": "Z.B.: Wie kann ich meine Bestellung verfolgen?",
        "send_button": "📤 Senden",
        "new_conversation": "🔄 Neues Gespräch",
        "export_chat": "📄 Chat Exportieren",
        "download_chat": "💾 Chat Herunterladen",
        "chat_controls": "💬 Chat-Steuerung",
        "chat_stats": "📊 Chat-Statistiken",
        "total_messages": "Gesamte Nachrichten",
        "your_questions": "Ihre Fragen",
        "information": "ℹ️ Information",
        "info_text": "Dieser Chatbot verwendet das Llama 3.1 8B Modell, das lokal über Ollama läuft.",
        "thinking": "🤖 Chatbot denkt nach...",
        "start_conversation": "👋 Beginnen Sie ein Gespräch, indem Sie eine Frage stellen!",
        "helpful_response": "Hilfreiche Antwort",
        "not_helpful": "Nicht hilfreich",
        "feedback_positive": "Danke für Ihr Feedback!",
        "feedback_negative": "Danke für Ihr Feedback. Wir werden uns verbessern!",
        "export_header": "Kundensupport Chat Export"
    },
    "🇮🇹 Italiano": {
        "code": "it",
        "name": "Italian",
        "title": "🤖 Chatbot di Supporto Clienti",
        "subtitle": "💬 Fai la tua domanda sui nostri prodotti o servizi. Il nostro assistente IA risponderà professionalmente e ricorderà la nostra conversazione.",
        "input_placeholder": "Es.: Come posso tracciare il mio ordine?",
        "send_button": "📤 Invia",
        "new_conversation": "🔄 Nuova Conversazione",
        "export_chat": "📄 Esporta Chat",
        "download_chat": "💾 Scarica Chat",
        "chat_controls": "💬 Controlli Chat",
        "chat_stats": "📊 Statistiche Chat",
        "total_messages": "Messaggi Totali",
        "your_questions": "Le Tue Domande",
        "information": "ℹ️ Informazioni",
        "info_text": "Questo chatbot usa il modello Llama 3.1 8B in esecuzione localmente tramite Ollama.",
        "thinking": "🤖 Il chatbot sta pensando...",
        "start_conversation": "👋 Inizia una conversazione facendo una domanda!",
        "helpful_response": "Risposta utile",
        "not_helpful": "Non utile",
        "feedback_positive": "Grazie per il tuo feedback!",
        "feedback_negative": "Grazie per il tuo feedback. Miglioreremo!",
        "export_header": "Esportazione Chat Supporto Clienti"
    }
}

st.set_page_config(
    page_title="Multilingual Customer Support Chatbot", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "history" not in st.session_state:
    st.session_state.history = []

if "selected_language" not in st.session_state:
    st.session_state.selected_language = "🇺🇸 English"

# NEW: Initialize input control for clearing
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

def get_current_lang_config():
    """Get current language configuration"""
    return LANGUAGES[st.session_state.selected_language]

def build_conversation_prompt(history, current_query):
    """Build contextualized prompt with conversation history and language support"""
    lang_config = get_current_lang_config()
    language_name = lang_config["name"]
    
    prompt = f"""You are a professional multilingual customer support chatbot. Always respond in {language_name} language.
Provide helpful, accurate, and concise responses. Remember the conversation context and refer to previous messages when relevant.

Guidelines:
- Always respond in {language_name}
- Be professional and courteous
- Provide specific and actionable solutions
- If you don't know something, admit it politely
- Keep responses concise but complete

Conversation History:
"""
    
    # Include last 10 messages for context (avoid token limit)
    recent_history = history[-10:] if len(history) > 10 else history
    
    for msg in recent_history:
        if msg["role"] == "user":
            prompt += f"Customer: {msg['content']}\n"
        else:
            prompt += f"Support: {msg['content']}\n"
    
    prompt += f"Customer: {current_query}\nSupport:"
    return prompt

def get_chatbot_response(user_query, max_retries=2):
    """Get response from Ollama with retry logic and error handling"""
    lang_config = get_current_lang_config()
    
    for attempt in range(max_retries + 1):
        try:
            prompt = build_conversation_prompt(st.session_state.history, user_query)
            
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME, 
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 512,
                        "stop": ["Customer:", "User:", "Cliente:", "Utilisateur:", "Kunde:", "Utente:"]
                    }
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                try:
                    json_response = json.loads(response.text)
                    bot_response = json_response.get("response", "").strip()
                    
                    if not bot_response:
                        fallback_messages = {
                            "en": "I apologize, but I couldn't generate a proper response. Could you please rephrase your question?",
                            "es": "Me disculpo, pero no pude generar una respuesta adecuada. ¿Podrías reformular tu pregunta?",
                            "fr": "Je m'excuse, mais je n'ai pas pu générer une réponse appropriée. Pourriez-vous reformuler votre question?",
                            "de": "Entschuldigung, aber ich konnte keine angemessene Antwort generieren. Könnten Sie Ihre Frage umformulieren?",
                            "it": "Mi scuso, ma non sono riuscito a generare una risposta appropriata. Potresti riformulare la tua domanda?"
                        }
                        return fallback_messages.get(lang_config["code"], fallback_messages["en"])
                    
                    return bot_response
                    
                except json.JSONDecodeError:
                    if attempt < max_retries:
                        time.sleep(1)
                        continue
                    error_messages = {
                        "en": "❌ Error: Invalid response format from server.",
                        "es": "❌ Error: Formato de respuesta inválido del servidor.",
                        "fr": "❌ Erreur: Format de réponse invalide du serveur.",
                        "de": "❌ Fehler: Ungültiges Antwortformat vom Server.",
                        "it": "❌ Errore: Formato di risposta non valido dal server."
                    }
                    return error_messages.get(lang_config["code"], error_messages["en"])
            else:
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                error_messages = {
                    "en": f"❌ Server error (Status: {response.status_code}). Please try again.",
                    "es": f"❌ Error del servidor (Estado: {response.status_code}). Por favor intenta de nuevo.",
                    "fr": f"❌ Erreur serveur (Statut: {response.status_code}). Veuillez réessayer.",
                    "de": f"❌ Serverfehler (Status: {response.status_code}). Bitte versuchen Sie es erneut.",
                    "it": f"❌ Errore server (Stato: {response.status_code}). Per favore riprova."
                }
                return error_messages.get(lang_config["code"], error_messages["en"])
                
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                timeout_warnings = {
                    "en": f"⏱️ Timeout on attempt {attempt + 1}. Retrying...",
                    "es": f"⏱️ Tiempo agotado en intento {attempt + 1}. Reintentando...",
                    "fr": f"⏱️ Timeout à la tentative {attempt + 1}. Nouvelle tentative...",
                    "de": f"⏱️ Timeout bei Versuch {attempt + 1}. Wiederholung...",
                    "it": f"⏱️ Timeout al tentativo {attempt + 1}. Riprovando..."
                }
                st.warning(timeout_warnings.get(lang_config["code"], timeout_warnings["en"]))
                time.sleep(1)
                continue
            timeout_errors = {
                "en": "⏱️ Request timed out. Please check your connection and try again.",
                "es": "⏱️ Solicitud agotada. Por favor verifica tu conexión e intenta de nuevo.",
                "fr": "⏱️ Demande expirée. Veuillez vérifier votre connexion et réessayer.",
                "de": "⏱️ Anfrage-Timeout. Bitte überprüfen Sie Ihre Verbindung und versuchen Sie es erneut.",
                "it": "⏱️ Richiesta scaduta. Per favore controlla la tua connessione e riprova."
            }
            return timeout_errors.get(lang_config["code"], timeout_errors["en"])
            
        except requests.exceptions.ConnectionError:
            connection_errors = {
                "en": "🔌 Cannot connect to Ollama server. Please ensure Ollama is running on localhost:11434",
                "es": "🔌 No se puede conectar al servidor Ollama. Asegúrate de que Ollama esté ejecutándose en localhost:11434",
                "fr": "🔌 Impossible de se connecter au serveur Ollama. Assurez-vous qu'Ollama fonctionne sur localhost:11434",
                "de": "🔌 Kann nicht mit Ollama-Server verbinden. Stellen Sie sicher, dass Ollama auf localhost:11434 läuft",
                "it": "🔌 Impossibile connettersi al server Ollama. Assicurati che Ollama sia in esecuzione su localhost:11434"
            }
            return connection_errors.get(lang_config["code"], connection_errors["en"])
            
        except Exception as e:
            general_errors = {
                "en": f"❌ Unexpected error: {str(e)}",
                "es": f"❌ Error inesperado: {str(e)}",
                "fr": f"❌ Erreur inattendue: {str(e)}",
                "de": f"❌ Unerwarteter Fehler: {str(e)}",
                "it": f"❌ Errore inaspettato: {str(e)}"
            }
            return general_errors.get(lang_config["code"], general_errors["en"])
    
    final_errors = {
        "en": "❌ Failed after multiple attempts. Please try again later.",
        "es": "❌ Falló después de múltiples intentos. Por favor intenta más tarde.",
        "fr": "❌ Échec après plusieurs tentatives. Veuillez réessayer plus tard.",
        "de": "❌ Nach mehreren Versuchen fehlgeschlagen. Bitte versuchen Sie es später erneut.",
        "it": "❌ Fallito dopo più tentativi. Per favore riprova più tardi."
    }
    return final_errors.get(lang_config["code"], final_errors["en"])

def display_chat_history():
    """Display conversation history with modern UI"""
    lang_config = get_current_lang_config()
    
    if not st.session_state.history:
        st.info(lang_config["start_conversation"])
        return
    
    # Create scrollable chat container
    chat_container = st.container()
    
    with chat_container:
        for i, msg in enumerate(st.session_state.history):
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div style='text-align: right; margin: 10px 0;'>
                        <div style='display: inline-block; background-color: #dcf8c6; 
                        padding: 10px 15px; border-radius: 15px 15px 5px 15px; 
                        max-width: 80%; box-shadow: 0 1px 2px rgba(0,0,0,0.1);'>
                            <strong>You:</strong> {msg['content']}
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style='text-align: left; margin: 10px 0;'>
                        <div style='display: inline-block; background-color: #f1f1f1; 
                        padding: 10px 15px; border-radius: 15px 15px 15px 5px; 
                        max-width: 80%; box-shadow: 0 1px 2px rgba(0,0,0,0.1);'>
                            <strong>🤖 Support:</strong> {msg['content']}
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Add rating buttons for bot messages (only latest)
                if i == len(st.session_state.history) - 1:
                    col1, col2, col3 = st.columns([1, 1, 6])
                    with col1:
                        if st.button("👍", key=f"like_{i}", help=lang_config["helpful_response"]):
                            st.session_state.history[i]["rating"] = "positive"
                            st.success(lang_config["feedback_positive"])
                    with col2:
                        if st.button("👎", key=f"dislike_{i}", help=lang_config["not_helpful"]):
                            st.session_state.history[i]["rating"] = "negative"
                            st.error(lang_config["feedback_negative"])

# Get current language configuration
lang_config = get_current_lang_config()

# Main UI
st.title(lang_config["title"])
st.markdown(lang_config["subtitle"])

# Sidebar for controls and info
with st.sidebar:
    st.header("🌍 Language Selection")
    
    # Simple language selection - no auto-detect
    selected_lang = st.selectbox(
        "Choose Interface Language:",
        options=list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(st.session_state.selected_language),
        help="Select your preferred language for the interface and chatbot responses"
    )
    
    # Update language if changed
    if selected_lang != st.session_state.selected_language:
        st.session_state.selected_language = selected_lang
        st.success(f"✅ Language changed to {selected_lang}")
        st.rerun()
    
    st.divider()
    st.header(lang_config["chat_controls"])
    
    if st.button(lang_config["new_conversation"], help="Clear chat history"):
        st.session_state.history = []
        # Also reset input field
        st.session_state.input_key += 1
        st.rerun()
    
    if st.session_state.history:
        st.subheader(lang_config["chat_stats"])
        total_messages = len(st.session_state.history)
        user_messages = len([m for m in st.session_state.history if m["role"] == "user"])
        st.metric(lang_config["total_messages"], total_messages)
        st.metric(lang_config["your_questions"], user_messages)
        
        # Export functionality
        if st.button(lang_config["export_chat"]):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chat_export = f"{lang_config['export_header']} - {timestamp}\n" + "="*50 + "\n\n"
            
            for msg in st.session_state.history:
                role = "Customer" if msg["role"] == "user" else "Support Agent"
                chat_export += f"{role}: {msg['content']}\n\n"
            
            st.download_button(
                lang_config["download_chat"],
                chat_export,
                file_name=f"support_chat_{lang_config['code']}_{timestamp}.txt",
                mime="text/plain"
            )
    
    st.subheader(lang_config["information"])
    st.info(lang_config["info_text"])

# FIXED: Main chat interface with proper input clearing
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "💭 " + lang_config["input_placeholder"].replace("E.g.: ", "").replace("Ej.: ", "").replace("Ex.: ", "").replace("Z.B.: ", "").replace("Es.: ", ""), 
        placeholder=lang_config["input_placeholder"],
        help="Ask any question about our products or services",
        key=f"user_input_{st.session_state.input_key}"
    )
    
    # Send button
    submitted = st.form_submit_button(
        lang_config["send_button"], 
        type="primary",
        use_container_width=True
    )

# Handle form submission
if submitted and user_input.strip():
    # Manage history size
    if len(st.session_state.history) >= MAX_HISTORY:
        st.session_state.history = st.session_state.history[-(MAX_HISTORY-2):]
    
    # Add user message
    st.session_state.history.append({
        "role": "user", 
        "content": user_input.strip(),
        "timestamp": datetime.now().isoformat(),
        "language": lang_config["code"]
    })
    
    # Generate and add bot response
    with st.spinner(lang_config["thinking"]):
        bot_reply = get_chatbot_response(user_input.strip())
        st.session_state.history.append({
            "role": "bot", 
            "content": bot_reply,
            "timestamp": datetime.now().isoformat(),
            "language": lang_config["code"]
        })
    
    # Update input key to ensure field stays cleared
    st.session_state.input_key += 1
    st.rerun()

# Display the conversation
display_chat_history()

# Enhanced CSS styling
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 20px;
        border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    /* Form styling */
    .stForm {
        border: none;
        padding: 0;
    }
    
    .stForm > div {
        gap: 0.5rem;
    }
    
    /* Custom scrollbar for chat */
    .main .block-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .main .block-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .main .block-container::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    
    .main .block-container::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """,
    unsafe_allow_html=True
)