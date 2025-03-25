import os
import torch

from langchain.vectorstores import FAISS
from Loader.load_pdf import PDFLoader
from Loader.embedding import EmbeddingGenerator

# Path to the folder containing your PDFs
pdf_folder = "./DatasetFinal"

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

# Extract text and metadata from each chunk.
# Each chunk is a dict with keys "content" and "metadata"
text_chunks = [chunk["content"] for chunk in chunks]
metadatas = [chunk["metadata"] for chunk in chunks]

print("Generating embeddings...")

api_key = "CPHwxBTkpGr5svldVyrUr1aL21NgDDj7"
# model_name = "mistral-embed"
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


generator = EmbeddingGenerator(api_key=api_key, model_name=model_name)
faiss_index_dir = "faiss_index"
generator.save_faiss_index(text_chunks, metadatas, faiss_index_dir)
