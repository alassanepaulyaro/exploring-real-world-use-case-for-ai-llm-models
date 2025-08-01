# AI Virtual Assistant
import streamlit as st
import requests
import json
import datetime

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"  # You can use "llama3.1:8b" or any model available in Ollama

st.set_page_config(page_title="AI Virtual Assistant", layout="centered")

st.title("🤖 AI Virtual Assistant")
st.markdown(
    "Ask me to schedule a task or any question! "
    "If you ask to schedule or remind, your task will be saved below."
)

# Initialize conversation history and tasks
if "history" not in st.session_state:
    st.session_state["history"] = []
if "scheduled_tasks" not in st.session_state:
    st.session_state["scheduled_tasks"] = []

def get_assistant_response(user_query):
    headers = {"Content-Type": "application/json"}
    prompt = (
        "You are an AI-powered virtual assistant that helps with task scheduling and answering queries.\n"
        "If the user asks to schedule a task, extract the task details and save it.\n"
        f"User: {user_query}\n"
        "Assistant:"
    )
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            headers=headers,
            timeout=90,
        )
        response_data = response.text.strip()
        try:
            json_response = json.loads(response_data)
        except json.JSONDecodeError:
            return "Error: Invalid JSON response from Ollama.", False
        assistant_reply = json_response.get("response", "I'm sorry, but I couldn't generate a response.")

        # Detect if the query is for scheduling
        scheduled = False
        if "schedule" in user_query.lower() or "remind" in user_query.lower():
            task = {
                "task": user_query,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state["scheduled_tasks"].append(task)
            assistant_reply += f"\n🗓️ **Task Scheduled:** {user_query}"
            scheduled = True

        return assistant_reply, scheduled
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {e}", False

# User input
user_input = st.text_input(
    "What can I help you with?",
    key="user_input",
    placeholder="E.g.: Schedule a meeting for tomorrow at 2pm"
)

if st.button("Send") and user_input:
    st.session_state["history"].append({"role": "user", "content": user_input})
    with st.spinner("Assistant is generating a response..."):
        assistant_reply, scheduled = get_assistant_response(user_input)
        st.session_state["history"].append({"role": "assistant", "content": assistant_reply})
    st.rerun()

# Show conversation history
st.subheader("Conversation")
if st.session_state["history"]:
    for msg in st.session_state["history"]:
        if msg["role"] == "user":
            st.markdown(f"<div style='background-color:#e6f0fa;padding:8px;border-radius:8px;margin-bottom:2px'><b>🧑 You:</b> {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color:#f6f6f6;padding:8px;border-radius:8px;margin-bottom:10px'><b>🤖 Assistant:</b> {msg['content']}</div>", unsafe_allow_html=True)

# Show scheduled tasks
st.subheader("📅 Scheduled Tasks")
if st.session_state["scheduled_tasks"]:
    for idx, task in enumerate(st.session_state["scheduled_tasks"], 1):
        st.markdown(f"**{idx}.** {task['task']}  \n<small><i>{task['timestamp']}</i></small>", unsafe_allow_html=True)
else:
    st.info("No tasks scheduled yet.")

# Option to clear everything
if st.button("🔄 New session"):
    st.session_state["history"] = []
    st.session_state["scheduled_tasks"] = []
    st.rerun()