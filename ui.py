import streamlit as st
import pandas as pd


# Function to simulate interaction with an LLM
def get_llm_response(user_input):
    # Placeholder for LLM interaction
    return f"LLM response to: {user_input}"


# Streamlit app layout
st.title("LLM Chat Interface")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("You:", key="user_input")

# Display chat history
for chat in st.session_state.chat_history:
    st.write(f"You: {chat['user']}")
    st.write(f"LLM: {chat['llm']}")

# Handle user input
if user_input:
    llm_response = get_llm_response(user_input)
    st.session_state.chat_history.append({"user": user_input, "llm": llm_response})
    st.experimental_rerun()
