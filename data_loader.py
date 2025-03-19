<<<<<<< HEAD
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import torch
import os
from pypdf import PdfReader
import fitz



# Function to load PDFs using fitz (PyMuPDF)
def load_pdfs_with_fitz(folder_path):
    documents = []
    for pdf_file in os.listdir(folder_path):
        if pdf_file.endswith(".pdf"):  # Ensure it's a PDF
            pdf_path = os.path.join(folder_path, pdf_file)
            try:
                pdf_document = fitz.open(pdf_path)
                print(f"✔ Successfully opened '{pdf_file}'. It has {pdf_document.page_count} pages.")
                
                # Extract text from each page
                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    text = page.get_text()  # Extract text from the page
                    # Append as a LangChain document
                    documents.append(
                        Document(page_content=text, metadata={"file_name": pdf_file, "page": page_num + 1})
                    )
                pdf_document.close()
            except Exception as e:
                print(f"⚠ Error reading '{pdf_file}': {e}")
    return documents










# # Iterate through all files in the folder to see formatting problems
# for pdf_file in os.listdir(pdf_folder):
#     if pdf_file.endswith(".pdf"):  # Ensure it's a PDF
#         pdf_path = os.path.join(pdf_folder, pdf_file)
#         try:
#             # Load the PDF file
#             reader = PdfReader(pdf_path)
            
#             # Check if the PDF is encrypted
#             if reader.is_encrypted:
#                 print(f" The PDF '{pdf_file}' is encrypted.")
#                 # Attempt decryption (replace 'password' with the actual password if known)
#                 try:
#                     reader.decrypt("password")
#                     print(f" Successfully decrypted '{pdf_file}'.")
#                 except Exception as decrypt_error:
#                     print(f" Failed to decrypt '{pdf_file}': {decrypt_error}")
#             else:
#                 print(f" The PDF '{pdf_file}' is not encrypted.")
#         except PdfReadError as e:
#             print(f" Error reading '{pdf_file}': {e}")
#         except Exception as generic_error:
#             print(f" An unexpected error occurred with '{pdf_file}': {generic_error}")





# Path to the folder containing your PDFs
pdf_folder = "./Dataset/"

# Check if GPU is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")



# Specify the path to your PDF file
print(device)

print("a")
# loader = DirectoryLoader(
#     "./Dataset/", glob="./*.pdf", loader_cls=PyPDFLoader
# )
print("b")
print("Loading PDFs...")
documents = load_pdfs_with_fitz(pdf_folder)

# loader_one_doc = DirectoryLoader(
#     "/1-EPC_17th_edition_2020_en.pdf",
#     glob="./*.pdf",
#     loader_cls=UnstructuredPDFLoader,
# )
# documents = loader.load()
print("c")
=======
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
>>>>>>> refs/remotes/origin/DATA

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
<<<<<<< HEAD
print("g")


=======
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
>>>>>>> refs/remotes/origin/DATA
