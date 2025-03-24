import os
import torch
from mistralai import Mistral
from langchain.vectorstores import FAISS
from Loader.load_pdf import PDFLoader

# Path to the folder containing your PDFs
pdf_folder = "./Dataset/"

# Check if GPU is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

print("Start")

print("Loading PDFs...")
loader = PDFLoader()
chunks = loader.load_dataset(pdf_folder, chunk_size=14000, overlap=1000)
print(len(chunks))

# Initialize Mistral client
api_key = "CPHwxBTkpGr5svldVyrUr1aL21NgDDj7"
model_name = "mistral-embed"
client = Mistral(api_key=api_key)


def generate_embeddings(chunks):
    """
    Generate embeddings for the given chunks of text using Mistral API.
    """
    chunk_texts = [chunk.page_content for chunk in chunks]
    embeddings_batch_response = client.embeddings.create(
        model=model_name,
        inputs=chunk_texts,
    )
    chunk_vectors = embeddings_batch_response["embeddings"]
    return chunk_vectors, embeddings_batch_response


print("Generating embeddings for chunks...")
chunk_vectors, embeddings = generate_embeddings(chunks)
text_embeddings = list(zip([chunk.page_content for chunk in chunks], chunk_vectors))

faiss_index_dir = "faiss_index"
if os.path.exists(faiss_index_dir):
    # Load the existing FAISS index (embeddings generation is skipped)
    faiss_index = FAISS.load_local(
        faiss_index_dir, embeddings, allow_dangerous_deserialization=True
    )
    print("Loaded existing FAISS index from", faiss_index_dir)
else:
    # If index doesn't exist, generate embeddings and build the index
    print("Generating embeddings for chunks...")
    faiss_index = FAISS.from_embeddings(text_embeddings, embeddings)
    faiss_index.save_local(faiss_index_dir)
    print("FAISS index built and saved to", faiss_index_dir)

print("Finished")
