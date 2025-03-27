import sys
import os
import torch

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import random
from RAG.RAG import RAG
from streamlit.components.v1 import html

# Activate GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def get_rag_instance():
    return RAG(
        embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        faiss_index_dir="faiss_index",
        llm_model_name="gemma2:2b",
    )


class RAGInterface:
    def __init__(self):
        st.set_page_config(
            page_title="LegalRAG Assistant",
            layout="wide",
            initial_sidebar_state="collapsed",
        )

        # Initialize session state
        self._initialize_session_state()

        # Load UI components
        self._load_ui_components()

        # Initialize RAG system
        self.rag_system = self._initialize_rag()

    def _initialize_session_state(self):
        """Initialize all session state variables"""
        if "mode" not in st.session_state:
            st.session_state.mode = "question"
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "exam_questions" not in st.session_state:
            st.session_state.exam_questions = []
        if "exam_answers" not in st.session_state:
            st.session_state.exam_answers = []
        if "current_question_idx" not in st.session_state:
            st.session_state.current_question_idx = 0

    def _load_ui_components(self):
        """Load CSS and JavaScript components"""
        current_dir = os.path.dirname(__file__)

        # Load CSS
        css_path = os.path.join(current_dir, "interface.css")
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        # Load JavaScript
        js_path = os.path.join(current_dir, "interface.js")
        with open(js_path, "r") as f:
            html(f"<script>{f.read()}</script>", height=0)

    
    def _initialize_rag(_self):
        try:
            return get_rag_instance()
        except FileNotFoundError:
            st.error(
                "FAISS index not found. Please run main.py first to create the index."
            )
            st.stop()


    def run_rag(self, user_input):
        return self.rag_system.query(user_input)

    def run(self):
        # 🔒 Always use light theme
        st.markdown(
            "<style>html, body, .stApp { background-color: #ffffff !important; color: #000000 !important; }</style>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='question-mode'>", unsafe_allow_html=True)

        st.title("🧠 Bytelex v1")

        # switch button
        if st.button("Switch Mode", key="mode_switch"):
            st.session_state.mode = (
                "exam" if st.session_state.mode == "question" else "question"
            )
            st.rerun()

        st.subheader(f"Current Mode: {st.session_state.mode.title()}")

        if st.session_state.mode == "question":
            self.run_question_mode()
        else:
            self.run_exam_mode()

        st.markdown("</div>", unsafe_allow_html=True)

    def run_question_mode(self):
        # Wrap the text input in a form to control submission
        with st.form(key="question_form"):
            user_input = st.text_input("Ask your question:")
            submit_button = st.form_submit_button("Submit")
        
        # Process the query only when the user clicks the submit button
        if submit_button and user_input:
            with st.spinner("Thinking..."):
                answer = self.run_rag(user_input)
            if answer.startswith("Error:"):
                st.error(answer)
            else:
                st.session_state.chat_history.append(
                    {"user": user_input, "assistant": answer}
                )
        
        # Display the chat history
        for exchange in st.session_state.chat_history:
            st.markdown(f"**You:** {exchange['user']}")
            st.markdown(f"**Bot:** {exchange['assistant']}")
            st.markdown("---")


    def run_exam_mode(self):
        if len(st.session_state.exam_questions) < 10:
            if st.session_state.current_question_idx == len(
                st.session_state.exam_questions
            ):
                question = random.choice(self.rag_system.test_questions)
                st.session_state.exam_questions.append(question)
            else:
                question = st.session_state.exam_questions[
                    st.session_state.current_question_idx
                ]

            st.markdown(
                f"**Question {st.session_state.current_question_idx + 1}/10:** {question}"
            )
            answer = st.text_area("Your answer:")

            if st.button("Submit Answer"):
                st.session_state.exam_answers.append(answer)
                st.session_state.current_question_idx += 1
                st.rerun()
        else:
            st.success("🎉 Exam Completed!")
            st.markdown("### Your Answers:")
            for idx, (q, a) in enumerate(
                zip(st.session_state.exam_questions, st.session_state.exam_answers)
            ):
                st.markdown(f"**Q{idx + 1}:** {q}")
                st.markdown(f"**Your Answer:** {a}")
                st.markdown("---")

            if st.button("Reset Exam"):
                st.session_state.exam_questions = []
                st.session_state.exam_answers = []
                st.session_state.current_question_idx = 0
                st.rerun()


if __name__ == "__main__":
    app = RAGInterface()
    app.run()
