from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


import torch

# Check if GPU is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# Specify the path to your PDF file
print(device)
print("a")
loader = DirectoryLoader(
    "./Dataset_bis/", glob="./*.pdf", loader_cls=UnstructuredPDFLoader
)
print("b")
# documents = loader.load()
loader_one_doc = DirectoryLoader(
    "/1-EPC_17th_edition_2020_en.pdf",
    glob="./*.pdf",
    loader_cls=UnstructuredPDFLoader,
)
documents = loader.load()
print("c")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
print("d")
chunks = text_splitter.split_documents(documents)
print("e")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
print("f")
chunk_vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
print("g")
