import os
import torch
import faiss
from transformers import pipeline
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.docstore import InMemoryDocstore
from langchain.llms import HuggingFacePipeline
from langchain_mistralai import MistralAIEmbeddings

# Path to the folder containing the FAISS index
faiss_index_dir = "../Dataset/faiss_index"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

print("Start")

# Load the FAISS index
index = faiss.read_index(os.path.join(faiss_index_dir, "index.faiss"))

# Initialize the Mistral embeddings
embedding_model = MistralAIEmbeddings(model="mistral-embed")

# Initialize the FAISS vector store
docstore = InMemoryDocstore({})
index_to_docstore_id = {i: str(i) for i in range(index.ntotal)}
faiss_index = FAISS(
    embedding_function=embedding_model.embed_documents,
    docstore=docstore,
    index=index,
    index_to_docstore_id=index_to_docstore_id,
)

# Initialize the local language model
qa_pipeline = pipeline(
    "text-generation", model="distilgpt2", device=0 if torch.cuda.is_available() else -1
)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# Create the LLMChain
prompt_template = PromptTemplate(
    template="{context}\n\nQuestion: {question}\nAnswer:",
    input_variables=["context", "question"],
)
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# Create the RetrievalQA chain
retriever = faiss_index.as_retriever()
qa_chain = RetrievalQA(combine_documents_chain=llm_chain, retriever=retriever)


# Define a function to interact with the RAG system
def ask_question(query):
    response = qa_chain.run(query)
    return response


# Example interaction
if __name__ == "__main__":
    while True:
        query = input("Enter your question: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = ask_question(query)
        print(f"Response: {response}")
