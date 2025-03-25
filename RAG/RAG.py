# rag.py

from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceHub

class RAG:
    def __init__(self, embedding_model_name, faiss_index_dir, llm_model_name, huggingfacehub_api_token):
        # Initialize embedding model
        self.embedder = SentenceTransformer(embedding_model_name)

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

        # Few-shot examples for better prompting
        few_shot_examples = (
            "Context:\nA European patent application can be transferred from one company to another.\n\n"
            "Question: Can a European patent application be assigned to another company?\n"
            "Answer: Yes, a European patent application can be transferred to another company.\n"
            "---\n"
            "Context:\nSilver ions are known for their antibacterial properties and are used in textiles.\n\n"
            "Question: Why are silver ions used in yoga mats?\n"
            "Answer: Silver ions are used in yoga mats to reduce the growth of bacteria due to their antibacterial properties.\n"
            "---"
        )

        # Define enhanced prompt template with few-shot examples
        self.prompt_template = PromptTemplate(
            template=(
                "You are an expert legal assistant. Use the provided context to answer the user's question."
                " If the answer cannot be found in the context, say 'The provided context does not contain enough information.'\n\n"
                f"{few_shot_examples}\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
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


if __name__ == "__main__":
    import os

    # Setup
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    FAISS_INDEX_DIR = "path/to/your/faiss_index_dir"
    LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # Initialize RAG
    rag_system = RAG(
        embedding_model_name=EMBEDDING_MODEL,
        faiss_index_dir=FAISS_INDEX_DIR,
        llm_model_name=LLM_MODEL_NAME,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )

    # Get user input and query
    question = input("Enter your legal question: ")
    answer = rag_system.query(question)

    print("\nGenerated Answer:\n", answer)
