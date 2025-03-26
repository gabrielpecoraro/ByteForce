import streamlit as st


class RAGInterface:
    def __init__(self):
        st.set_page_config(page_title="RAG Interface", layout="wide")

        # Load and apply custom CSS
        with open("interface.css") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state variables"""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

    def run_rag(self, user_input):
        """
        Simulated RAG system response with mock legal document context and responses
        """
        import random
        import time

        # Simulate processing time
        time.sleep(1)

        # Mock document contexts
        contexts = {
            "EPC": "According to Article 52 EPC, European patents shall be granted for any inventions...",
            "PCT": "Rule 4 of the PCT states that the request shall contain the prescribed information...",
            "Guidelines": "The Guidelines for Examination specify that inventive step requires...",
        }

        # Simple keyword matching for demo
        response = "I apologize, but I couldn't find relevant information in the legal documents."

        keywords = {
            "patent": ["EPC", "PCT"],
            "invention": ["EPC"],
            "application": ["PCT"],
            "examination": ["Guidelines"],
        }

        # Find matching context
        for keyword, sources in keywords.items():
            if keyword.lower() in user_input.lower():
                source = random.choice(sources)
                context = contexts[source]
                confidence = random.uniform(0.7, 0.95)

                response = f"""Based on {source} documents (confidence: {confidence:.2f}):
                
{context}

This information comes from the {source} documentation."""
                break

        return response

    def display_chat_history(self):
        """Display the chat history with improved formatting"""
        for message in st.session_state.chat_history:
            with st.container():
                st.markdown(f"**🧑‍💻 User:** {message['user']}")
                st.markdown(f"**🤖 Assistant:** {message['assistant']}")
                st.markdown("---")

    def run(self):
        """Main interface method"""
        st.title("💬 LegalRAG: Patent Document Assistant")
        st.markdown("---")

        # Load and inject JavaScript from file
        with open("interface.js", "r") as file:
            js_code = file.read()
            st.components.v1.html(f"<script>{js_code}</script>", height=0)

        # Create the input text area with key for state management
        user_input = st.text_area(
            "Enter your question about patents (Press Enter to submit, Shift+Enter for new line):",
            height=100,
            key="input",
        )

        # Create the submit button with loading state
        if st.button("Submit", type="primary"):
            if user_input and len(user_input.strip()) > 0:
                with st.container():
                    st.markdown(
                        "<div class='thinking'>Thinking</div>", unsafe_allow_html=True
                    )
                    response = self.run_rag(user_input)
                    st.session_state.chat_history.append(
                        {"user": user_input, "assistant": response}
                    )
                    st.session_state.input = ""
                    st.rerun()

        # Display chat history
        self.display_chat_history()


if __name__ == "__main__":
    interface = RAGInterface()
    interface.run()
