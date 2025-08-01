# AI ecommerce recommendation 
import streamlit as st
import requests
import json
import pandas as pd

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "granite3.2"

# Sample Product Database
products = [
    {"id": 1, "category": "Electronics", "name": "Wireless Earbuds"},
    {"id": 2, "category": "Electronics", "name": "Smartphone"},
    {"id": 3, "category": "Electronics", "name": "Laptop"},
    {"id": 4, "category": "Fashion", "name": "Leather Jacket"},
    {"id": 5, "category": "Fashion", "name": "Running Shoes"},
    {"id": 6, "category": "Home", "name": "Smart Vacuum Cleaner"},
    {"id": 7, "category": "Home", "name": "Air Purifier"},
]
df = pd.DataFrame(products)

st.set_page_config(page_title="AI Product Recommender", layout="centered")
st.title("🤖 AI Product Recommender (Granite 3.2 + Ollama)")

st.markdown("""
Enter your preferences below.  
The AI will suggest the best matching products.  
""")

with st.form(key="recommend_form"):
    preferences = st.text_area("Describe your preferences", height=100, max_chars=500)
    submit = st.form_submit_button("Get Recommendations")

if submit and preferences.strip():
    with st.spinner("Getting product recommendations..."):
        # Optional: filter by categories if desired
        # Example: filter on 'Electronics', etc.

        prompt = f"""You are an AI product recommender. Based on the user's preferences, suggest the best matching products from this catalog:
{df[['category', 'name']].to_string(index=False)}

User Preferences: {preferences}

Recommended Products:"""

        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(
                OLLAMA_URL,
                json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
                headers=headers,
                timeout=60
            )
            response_data = response.text.strip()
            try:
                json_response = json.loads(response_data)
                ai_recommendations = json_response.get("response", "No recommendations found.")
            except Exception:
                st.error(f"Invalid response from Ollama: {response_data}")
                ai_recommendations = None

        except Exception as e:
            st.error(f"Request to Ollama failed: {e}")
            ai_recommendations = None

        if ai_recommendations:
            st.subheader("AI Recommendations")
            st.markdown(ai_recommendations)

    st.markdown("---")
    st.markdown("### Product Catalog")
    st.dataframe(df[["category", "name"]], use_container_width=True)

st.markdown("""
---
<small>Powered by Granite 3.2 + Ollama & Streamlit — Demo for product recommendation</small>
""", unsafe_allow_html=True)
