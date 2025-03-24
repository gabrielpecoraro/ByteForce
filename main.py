import os
import pickle
import string
import torch
import fitz
import nltk
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from pypdf import PdfReader

# import gensim
# import gensim.corpora as corpora
from nltk.corpus import stopwords
from Loader.load_pdf import PDFLoader


# Path to the folder containing your PDFs
pdf_folder = "./Dataset_bis/"


# Check if GPU is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Specify the path to your PDF file
print(device)

print("Start")


print("Loading PDFs...")
loader = PDFLoader()
chunks=loader.load_dataset(pdf_folder, chunk_size=14000, overlap=1000)
print(len(chunks))
