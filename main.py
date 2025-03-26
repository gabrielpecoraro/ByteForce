import torch
from Loader.load_pdf import PDFLoader
from Loader.embedding import EmbeddingGenerator
from RAG.RAG import RAG


# Path to the folder containing your PDFs
pdf_folder = "./Dataset_bis"

# Check if GPU is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Specify the path to your PDF file
print(device)

print("Start")

print("Loading PDFs...")
loader = PDFLoader()
# Example: Use a smaller chunk size
chunks = loader.load_dataset(pdf_folder)
print("Number of chunks:", len(chunks))

# Each chunk is a dict with keys "content" and "metadata"
text_chunks = [chunk["content"] for chunk in chunks]
metadatas = [chunk["metadata"] for chunk in chunks]
faiss_index_dir = "faiss_index"



api_key = "CPHwxBTkpGr5svldVyrUr1aL21NgDDj7"
# model_name = "mistral-embed"
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
llm_name="mistralai/Mistral-7B-Instruct-v0.2"

generator = EmbeddingGenerator(model_name=model_name)



# Save to FAISS using the real model
vectorstore = generator.save_embeddings_to_faiss(
    chunks=text_chunks,
    save_path="faiss_index",
    metadata_list=metadatas,
)
rag_system = RAG(
        embedding_model_name=model_name,
        faiss_index_dir=faiss_index_dir,
        llm_model_name=llm_name,
        huggingfacehub_api_token=api_key
    )

# === Ask a question interactively ===
print("\n✅ RAG system is ready.")

for q, _ in rag_system.test_questions:
    print(f"\n📌 Question: {q}")
    answer = rag_system.query(q)
    print("🧾 Answer:\n", answer)


