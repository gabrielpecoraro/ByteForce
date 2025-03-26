import streamlit as st
import streamlit.components.v1 as components
import os


class RAGInterface:
    def __init__(self):
        st.set_page_config(
            page_title="LegalRAG Assistant",
            layout="wide",
            initial_sidebar_state="collapsed",
        )

        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Load and inject CSS
        css_path = os.path.join(current_dir, "interface.css")
        with open(css_path, "r") as css_file:
            css = css_file.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

        # Load and inject JavaScript
        js_path = os.path.join(current_dir, "interface.js")
        with open(js_path, "r") as js_file:
            js = js_file.read()
            components.html(
                f"""
                <script>{js}</script>
                """,
                height=0,
            )

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

        # Create a container for chat history at the top
        chat_container = st.container()

        # Create input section at the bottom
        with st.container():
            # Initialize the input value from session state
            if "user_input" not in st.session_state:
                st.session_state.user_input = ""

            user_input = st.text_area(
                "Enter your question about patents (Press Enter to submit, Shift+Enter for new line):",
                value=st.session_state.user_input,
                height=100,
                key="input_area",
            )

            col1, col2 = st.columns([1, 5])

            with col1:
                if st.button("Submit", type="primary"):
                    if user_input and len(user_input.strip()) > 0:
                        with st.spinner("Thinking..."):
                            response = self.run_rag(user_input)
                            st.session_state.chat_history.append(
                                {"user": user_input, "assistant": response}
                            )
                            # Clear input by updating session state
                            st.session_state.user_input = ""
                            st.rerun()

            with col2:
                if st.button("Clear History", type="secondary"):
                    st.session_state.chat_history = []
                    st.rerun()

        # Display chat history in the container
        with chat_container:
            self.display_chat_history()


if __name__ == "__main__":
    interface = RAGInterface()
    interface.run()
