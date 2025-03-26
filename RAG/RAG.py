from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from docx import Document
import os
import random


def extract_qa_from_docx(path):
    doc = Document(path)
    qa_pairs = []
    question = ""
    answer = ""
    for para in doc.paragraphs:
        text = para.text.strip()
        if text.lower().startswith("question:"):
            question = text[len("question:"):].strip()
        elif text.lower().startswith("answer:"):
            answer = text[len("answer:"):].strip()
        elif text.strip() == "---" and question and answer:
            qa_pairs.append((question, answer))
            question, answer = "", ""
    if question and answer:
        qa_pairs.append((question, answer))
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


def get_random_test_questions(qa_pairs, n=5, seed=42):
    random.seed(seed)
    return random.sample(qa_pairs, min(n, len(qa_pairs)))


class RAG:
    def __init__(self, embedding_model_name, faiss_index_dir, llm_model_name, huggingfacehub_api_token):
        # Initialize embedding model
        self.embedder = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Load FAISS index
        self.vectorstore = FAISS.load_local(
            faiss_index_dir, self.embedder.encode, allow_dangerous_deserialization=True
        )

        # Initialize the LLM
        self.llm = HuggingFaceHub(
            huggingfacehub_api_token=huggingfacehub_api_token,
            repo_id=llm_model_name,
            model_kwargs={"temperature": 0.2, "max_length": 1024}
        )

        # Load and format few-shot examples
        qa_pairs = extract_qa_from_docx("Questions Sup OEB.docx")
        few_shot_examples = format_few_shots(qa_pairs, max_examples=8)
        self.test_questions = get_random_test_questions(qa_pairs, n=5, seed=42)  # repeatable sample

        # Define enhanced prompt template with few-shot examples
        self.prompt_template = PromptTemplate(
            template=(
                "You are an expert legal assistant trained to answer questions using legal context only.\n\n"
        "For each question, you are given a context extracted from legal sources. Answer only if the answer is explicitly present in the context.\n\n"
        "- Do not make up answers.\n"
        "- If the context is unclear or missing, say: \"The provided context does not contain enough information.\"\n"
        "- Format your answer clearly and concisely, like in the examples.\n\n"
                f"{few_shot_examples}\n"
                "Context:\n{{context}}\n\n"
                "Question: {{question}}\n\n"
                "Answer:"
            ),
            input_variables=["context", "question"]
        )

        # Define LLM chain
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def query(self, question, top_k=5):
        # Generate embedding for question
        question_embedding = self.embedder.encode([question])

        # Retrieve relevant context
        docs_and_scores = self.vectorstore.similarity_search_by_vector(question_embedding[0], k=top_k)
        context = "\n---\n".join([doc.page_content for doc in docs_and_scores])

        # Generate response from LLM
        response = self.chain.run({"context": context, "question": question})

        return response
