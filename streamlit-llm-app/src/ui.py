import streamlit as st
from llm_interface import query_llm

def main():
    st.title("Large Language Model Interface")
    
    user_input = st.text_area("Enter your query:")
    
    if st.button("Submit"):
        if user_input:
            response = query_llm(user_input)
            st.text_area("LLM Response:", value=response, height=300)
        else:
            st.warning("Please enter a query before submitting.")

if __name__ == "__main__":
    main()