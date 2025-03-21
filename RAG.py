import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

def load_faiss_index(faiss_index_dir, embeddings):
    if os.path.exists(faiss_index_dir):
        return FAISS.load_local(faiss_index_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        print("FAISS index directory not found.")
        return None

def advanced_retriever(query, faiss_index, k=10, chain_type="refine"):
    retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": k})
    gen_pipeline = pipeline("text-generation", model="gpt2", max_new_tokens=100)
    llm = HuggingFacePipeline(pipeline=gen_pipeline)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type=chain_type, retriever=retriever)
    return qa_chain.run(query)

faiss_index_dir = "faiss_index"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
faiss_index = load_faiss_index(faiss_index_dir, embeddings)

if faiss_index:
    query = "What is article 1"
    print("Query:", query)
    output = advanced_retriever(query, faiss_index, k=5, chain_type="refine")
    print("\nAdvanced RAG Output:\n", output)
else:
    print("No FAISS index available.")
