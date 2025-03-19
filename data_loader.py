from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from torch import embedding

# Specify the path to your PDF file

loader = DirectoryLoader('./Dataset/', glob="./*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = text_splitter.split_documents(documents)

# Extract text content from each chunk
texts = [chunk.page_content for chunk in chunks]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Generate embeddings for the list of text strings
chunk_vectors = embeddings.embed_documents(texts)








