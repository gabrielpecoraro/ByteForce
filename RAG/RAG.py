from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from docx import Document
import random
from huggingface_hub import InferenceClient


def extract_qa_from_docx(path):
    doc = Document(path)
    qa_pairs = []
    question = ""
    in_question = False
    answer = ""

    for para in doc.paragraphs:
        text = para.text.strip()

        if text.lower().startswith("question"):
            # Start of a new question
            in_question = True
            question = ""
            answer = ""
        elif text.lower().startswith("answer"):
            in_question = False  # Done collecting question
        elif text.lower().startswith("the correct answer is"):
            answer = text
            qa_pairs.append((question.strip(), answer.strip()))
        elif in_question:
            question += " " + text  # Append to current question

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
    import random

    random.seed(seed)
    return [q for q, _ in random.sample(qa_pairs, min(n, len(qa_pairs)))]


class RAG:
    def __init__(
        self,
        embedding_model_name,
        faiss_index_dir,
        llm_model_name,
        huggingfacehub_api_token,
    ):
        # Initialize embedding model
        self.embedder = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Load FAISS index
        self.vectorstore = FAISS.load_local(
            faiss_index_dir, self.embedder, allow_dangerous_deserialization=True
        )

        # Initialize Inference Client
        self.client = InferenceClient(
            model=llm_model_name, token=huggingfacehub_api_token
        )

        # Initialize test questions
        self.test_questions = [
            [
                "What are the requirements for patentability under EPC Article 32?",
            ]
        ]

        # Define prompt template
        self.prompt_template = PromptTemplate(
            template=(
                "You are an expert legal assistant trained to answer questions using legal context only.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
            input_variables=["context", "question"],
        )

    def query(self, question, top_k=5):
        # Get relevant documents
        docs = self.vectorstore.similarity_search(question, k=top_k)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Format prompt
        formatted_prompt = self.prompt_template.format(
            context=context, question=question
        )

        # Generate response using the newer API
        response = self.client.text_generation(
            formatted_prompt,
            max_new_tokens=512,
            temperature=0.5,
            do_sample=True,
            return_full_text=False,
        )

        return response
