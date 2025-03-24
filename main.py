import os
import torch
from mistralai import Mistral
from langchain.vectorstores import FAISS
from pypdf import PdfReader

# import gensim
# import gensim.corpora as corpora
from nltk.corpus import stopwords
from Loader.load_pdf import PDFLoader
from Loader.embedding import EmbeddingGenerator

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
chunks = loader.load_dataset(pdf_folder, chunk_size=512, overlap=100)
print("Number of chunks:", len(chunks))

# Extract text and metadata from each chunk.
# Each chunk is a dict with keys "content" and "metadata"
text_chunks = [chunk["content"] for chunk in chunks]
metadatas = [chunk["metadata"] for chunk in chunks]

print("Generating embeddings...")

api_key = "CPHwxBTkpGr5svldVyrUr1aL21NgDDj7"
model_name = "mistral-embed"
client = Mistral(api_key=api_key)
generator = EmbeddingGenerator(api_key=api_key)

# Generate embeddings for the text chunks.
# This function now gets a list of strings, not the full dict.
chunk_vectors = generator.generate_embeddings(text_chunks)
print("Saving embeddings.....")
# Create a list of tuples pairing each text with its metadata.
text_with_metadata = list(zip(text_chunks, metadatas))

faiss_index_dir = "faiss_index"
if os.path.exists(faiss_index_dir):
    # Load the existing FAISS index (using the stored embeddings)
    faiss_index = FAISS.load_local(
        faiss_index_dir, chunk_vectors, allow_dangerous_deserialization=True
    )
    print("Loaded existing FAISS index from", faiss_index_dir)
else:
    # Build the FAISS index using the text and metadata paired with embeddings.
    # The FAISS.from_embeddings method accepts a list of tuples (text, metadata) and their embeddings.
    faiss_index = FAISS.from_embeddings(text_with_metadata, chunk_vectors)
    faiss_index.save_local(faiss_index_dir)
    print("FAISS index built and saved to", faiss_index_dir)

print("Finished")

