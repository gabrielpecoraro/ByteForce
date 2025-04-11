import sys
import os
import torch
from codecarbon import OfflineEmissionsTracker

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import random
from RAG.RAG import RAG
from streamlit.components.v1 import html

#Activate GPU if available
device = torch.device("cuda" if torch.backends.mps.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_device(0)
    print("Using GPU")
else:
    print("Using CPU")


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
        if "response_mode" not in st.session_state:
            st.session_state.response_mode = "base"
        if "num_exam_questions" not in st.session_state:
            st.session_state.num_exam_questions = 10

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
        return self.rag_system.query(user_input, mode=st.session_state.response_mode)

    def run(self):
        # 🔒 Always use light theme
        st.markdown(
            "<style>html, body, .stApp { background-color: #ffffff !important; color: #000000 !important; }</style>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='question-mode'>", unsafe_allow_html=True)

        st.title("🧠 Bytelex v2")

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
        # Add response mode toggle at the top
        st.session_state.response_mode = "enhanced" if st.toggle(
            "Use Enhanced Response Mode",
            value=st.session_state.response_mode == "enhanced",
            help="Toggle between base and enhanced response modes"
        ) else "base"

        # Create a container for chat history that will be updated
        chat_container = st.container()

        # Wrap the text input in a form to control submission
        with st.form(key="question_form", clear_on_submit=True):
            user_input = st.text_input("Ask your question:")
            submit_button = st.form_submit_button("Submit")

        # Process the query only when the user clicks the submit button
        if submit_button and user_input:
            with st.spinner("Thinking..."):
                answer = self.run_rag(user_input)
            if answer.startswith("Error:"):
                st.error(answer)
            else:
                # Add new exchange to chat history
                st.session_state.chat_history.append({"user": user_input, "assistant": answer})

        # Display full chat history in reverse order
        with chat_container:
            for exchange in reversed(st.session_state.chat_history):
                with st.container():
                    st.markdown("---")
                    st.markdown(f"**You:** {exchange['user']}")
                    st.markdown(f"**Assistant:** {exchange['assistant']}")

    def run_exam_mode(self):
        # Add number of questions slider at the top
        st.session_state.num_exam_questions = st.slider(
            "Number of Questions",
            min_value=1,
            max_value=50,
            value=st.session_state.num_exam_questions,
            help="Select the number of questions for your exam"
        )

        # Generate exam questions until reaching the selected number
        if len(st.session_state.exam_questions) < st.session_state.num_exam_questions:
            if st.session_state.current_question_idx == len(st.session_state.exam_questions):
                question = self.rag_system.generate_exam_question()
                st.session_state.exam_questions.append(question)
            else:
                question = st.session_state.exam_questions[st.session_state.current_question_idx]

            st.markdown(
                f"**Question {st.session_state.current_question_idx + 1}/{st.session_state.num_exam_questions}:** {question}"
            )
            answer = st.text_area("Your answer:")

            if st.button("Submit Answer"):
                # Store the answer
                st.session_state.exam_answers.append(answer)
                # Get immediate feedback
                evaluation = self.rag_system.evaluate_answer(question, answer)
                st.session_state.current_question_idx += 1
                
                # Show feedback for the current answer
                st.markdown("### Feedback for your answer:")
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Your Answer:** {answer}")
                st.markdown(f"**Evaluation:** {evaluation}")
                st.markdown("---")
                
                if st.session_state.current_question_idx < st.session_state.num_exam_questions:
                    st.rerun()

        else:
            st.success("🎉 Exam Completed!")
            st.markdown("### Complete Exam Evaluation:")
            # Display all questions and evaluations
            for idx, (question, answer) in enumerate(zip(
                st.session_state.exam_questions,
                st.session_state.exam_answers
            )):
                evaluation = self.rag_system.evaluate_answer(question, answer)
                st.markdown(f"**Question {idx + 1}:** {question}")
                st.markdown(f"**Your Answer:** {answer}")
                st.markdown(f"**Evaluation:** {evaluation}")
                st.markdown("---")

            if st.button("Reset Exam"):
                st.session_state.exam_questions = []
                st.session_state.exam_answers = []
                st.session_state.current_question_idx = 0
                st.rerun()


if __name__ == "__main__":
    app = RAGInterface()
    app.run()
