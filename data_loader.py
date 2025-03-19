from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import os
import sqlite3

# Check if MPS (Metal Performance Shaders) is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Specify the path to your PDF files directory
pdf_directory = "./Dataset/"

# Load all PDF files from the directory
loader = DirectoryLoader(pdf_directory, glob="*.pdf", loader_cls=UnstructuredPDFLoader)
documents = loader.load()
print("Loaded documents")

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print("Split documents into chunks")

# Initialize the embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
print("Initialized embeddings model")

# Generate embeddings for each chunk
chunk_vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
print("Generated embeddings for chunks")

# Move embeddings to MPS device
chunk_vectors = [torch.tensor(vector).to(device) for vector in chunk_vectors]

# Store the embeddings in a database (e.g., SQLite)
# Create a new SQLite database (or connect to an existing one)
conn = sqlite3.connect("embeddings.db")
c = conn.cursor()

# Create a table to store the embeddings
c.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY,
        content TEXT,
        embedding BLOB
    )
""")

# Insert the embeddings into the database
for chunk, vector in zip(chunks, chunk_vectors):
    c.execute(
        "INSERT INTO embeddings (content, embedding) VALUES (?, ?)",
        (chunk.page_content, vector.cpu().numpy().tobytes()),
    )

# Commit the changes and close the connection
conn.commit()
conn.close()
print("Stored embeddings in the database")
