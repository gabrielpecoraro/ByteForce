from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from docx import Document
import random
from pathlib import Path
from RAG.llm_ollama import OllamaLLM


def extract_qa_from_docx(path):
    doc = Document(path)
    qa_pairs = []
    question = ""
    in_question = False
    answer = ""

    for para in doc.paragraphs:
        text = para.text.strip()

        if text.lower().startswith("question"):
            in_question = True
            question = ""
            answer = ""
        elif text.lower().startswith("answer"):
            in_question = False
        elif text.lower().startswith("the correct answer is"):
            answer = text
            qa_pairs.append((question.strip(), answer.strip()))
        elif in_question:
            question += " " + text

    return qa_pairs


def format_few_shots(qa_pairs, max_examples=3):
    prompt = ""
    for q, a in qa_pairs[:max_examples]:
        prompt += (
            "Context:\nThis is the type of legal information relevant to the question.\n\n"
            f"Question: {q}\n"
            f"Answer: {a}\n"
            "---\n"
        )
    return prompt


def get_random_test_questions(qa_pairs, n=5, seed=19):
    random.seed(seed)
    return [q for q, _ in random.sample(qa_pairs, min(n, len(qa_pairs)))]


class RAG:
    def __init__(
        self,
        embedding_model_name,
        faiss_index_dir,
        llm_model_name,
        few_shot_docx_path="Questions Sup OEB.docx",
        max_few_shot_examples=5,
    ):
        self.embedder = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vectorstore = FAISS.load_local(
            faiss_index_dir, self.embedder, allow_dangerous_deserialization=True
        )
        self.client = OllamaLLM(model_name=llm_model_name)


        self.few_shot_prompt = ""
        if few_shot_docx_path and Path(few_shot_docx_path).exists():
            qa_pairs = extract_qa_from_docx(few_shot_docx_path)
            self.few_shot_prompt = format_few_shots(qa_pairs, max_examples=max_few_shot_examples)
        self.test_questions = [
                "What are the requirements for patentability under EPC Article 52?"
            ]

        self.prompt_template = PromptTemplate(
            template=(
                "{few_shots}"
                "You are an expert legal assistant trained to answer questions using legal context only.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
            input_variables=["few_shots", "context", "question"],
        )

    def query(self, question, top_k=5):
        docs = self.vectorstore.similarity_search(question, k=top_k)
        context = "\n\n".join([doc.page_content for doc in docs])

        formatted_prompt = self.prompt_template.format(
            few_shots=self.few_shot_prompt,
            context=context,
            question=question,
        )

        response = self.client.generate(
            formatted_prompt,
            temperature=0.2,
            max_tokens=512,
        )


        return response
