import sys
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import random
from RAG.RAG import RAG
from streamlit.components.v1 import html

class RAGInterface:
    def __init__(self):
        st.set_page_config(
            page_title="AI Assistant",
            layout="wide",
            initial_sidebar_state="collapsed",
        )

        # Inject CSS and JS for UI enhancements
        css_path = os.path.join(os.path.dirname(__file__), "interface.css")
        js_path = os.path.join(os.path.dirname(__file__), "interface.js")

        with open(css_path, "r") as f:
            styles = f"<style>{f.read()}</style>"
            st.markdown(styles, unsafe_allow_html=True)

        with open(js_path, "r") as f:
            script = f"<script>{f.read()}</script>"
            html(script, height=0)

        # Set default mode to "question" if not already set
        if "mode" not in st.session_state:
            st.session_state.mode = "question"

        # Initialize RAG backend only once
        self.rag_system = self._initialize_rag()

        # Session vars
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "exam_questions" not in st.session_state:
            st.session_state.exam_questions = []
        if "exam_answers" not in st.session_state:
            st.session_state.exam_answers = []
        if "current_question_idx" not in st.session_state:
            st.session_state.current_question_idx = 0

    #@st.cache_resource(show_spinner=False)
    def _initialize_rag(_self):
        return RAG(
            embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            faiss_index_dir="faiss_index",
            llm_model_name="mistral"
        )

    def run_rag(self, user_input):
        return self.rag_system.query(user_input)

    def run(self):
        # 🔒 Always use light theme
        st.markdown("<style>html, body, .stApp { background-color: #ffffff !important; color: #000000 !important; }</style>", unsafe_allow_html=True)
        st.markdown("<div class='question-mode'>", unsafe_allow_html=True)

        st.title("🧠 Bytelex v1")

        # ✅ Restore mode switch functionality
        if st.button("Switch Mode", key="mode_switch"):
            st.session_state.mode = "exam" if st.session_state.mode == "question" else "question"
            st.rerun()

        st.subheader(f"Current Mode: {st.session_state.mode.title()}")

        if st.session_state.mode == "question":
            self.run_question_mode()
        else:
            self.run_exam_mode()

        st.markdown("</div>", unsafe_allow_html=True)

    def run_question_mode(self):
        user_input = st.text_input("Ask your question:")
        if user_input:
            with st.spinner("Thinking..."):
                answer = self.run_rag([user_input])[0]
            st.session_state.chat_history.append({"user": user_input, "assistant": answer})
            st.rerun()

        for exchange in st.session_state.chat_history:
            st.markdown(f"**You:** {exchange['user']}")
            st.markdown(f"**Bot:** {exchange['assistant']}")
            st.markdown("---")

    def run_exam_mode(self):
        if len(st.session_state.exam_questions) < 10:
            if st.session_state.current_question_idx == len(st.session_state.exam_questions):
                question = random.choice(self.rag_system.test_questions)
                st.session_state.exam_questions.append(question)
            else:
                question = st.session_state.exam_questions[st.session_state.current_question_idx]

            st.markdown(f"**Question {st.session_state.current_question_idx + 1}/10:** {question}")
            answer = st.text_area("Your answer:")

            if st.button("Submit Answer"):
                st.session_state.exam_answers.append(answer)
                st.session_state.current_question_idx += 1
                st.rerun()
        else:
            st.success("🎉 Exam Completed!")
            st.markdown("### Your Answers:")
            for idx, (q, a) in enumerate(zip(st.session_state.exam_questions, st.session_state.exam_answers)):
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
