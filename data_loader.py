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
from langchain.vectorstores import FAISS
import pickle
# from langchain.llms import


# Function to load PDFs using fitz (PyMuPDF)
def load_pdfs_with_fitz(folder_path):
    documents = []
    for pdf_file in os.listdir(folder_path):
        if pdf_file.endswith(".pdf"):  # Ensure it's a PDF
            pdf_path = os.path.join(folder_path, pdf_file)
            try:
                pdf_document = fitz.open(pdf_path)
                print(
                    f"✔ Successfully opened '{pdf_file}'. It has {pdf_document.page_count} pages."
                )

                # Extract text from each page
                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    text = page.get_text()  # Extract text from the page
                    # Append as a LangChain document
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"file_name": pdf_file, "page": page_num + 1},
                        )
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
pdf_folder = "./Dataset_bis/"

# Check if GPU is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Specify the path to your PDF file
print(device)

print("Start")

#
# loader = DirectoryLoader(
#     "./Dataset/", glob="./*.pdf", loader_cls=PyPDFLoader
# )

print("Loading PDFs...")
documents_file = "documents.pkl"
if os.path.exists(documents_file):
    with open(documents_file, "rb") as f:
        documents = pickle.load(f)
    print("Loaded cached documents from", documents_file)
else:
    print("Loading PDFs from folder...")
    documents = load_pdfs_with_fitz(pdf_folder)
    with open(documents_file, "wb") as f:
        pickle.dump(documents, f)
    print("Documents loaded and cached to", documents_file)

# loader_one_doc = DirectoryLoader(
#     "/1-EPC_17th_edition_2020_en.pdf",
#     glob="./*.pdf",
#     loader_cls=UnstructuredPDFLoader,
# )
# documents = loader.load()
print("splitting pdfs...")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

chunks = text_splitter.split_documents(documents)
print("embedding...")

faiss_index_dir = "faiss_index"
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

if os.path.exists(faiss_index_dir):
    # Load the existing FAISS index (embeddings generation is skipped)
    faiss_index = FAISS.load_local(
        faiss_index_dir, embeddings, allow_dangerous_deserialization=True
    )
    print("Loaded existing FAISS index from", faiss_index_dir)
else:
    # If index doesn't exist, generate embeddings and build the index
    print("Generating embeddings for chunks...")
    chunk_vectors = embeddings.embed_documents(
        [chunk.page_content for chunk in chunks]
    )  # should use gpu
    text_embeddings = list(zip([chunk.page_content for chunk in chunks], chunk_vectors))

    # Note: The following line computes embeddings and builds the FAISS index
    faiss_index = FAISS.from_embeddings(text_embeddings, embeddings)

    faiss_index.save_local(faiss_index_dir)
    print("FAISS index built and saved to", faiss_index_dir)

print("finish")


print("queries")

query = "What is question 1"
query_vector = embeddings.embed_query(query)
results = faiss_index.similarity_search_by_vector(
    query_vector, k=5
)  # `k` is the number of top matches
for result in results:
    print(result.page_content)  # This shows the most relevant text chunks
