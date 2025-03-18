from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Specify the path to your PDF file

loader = DirectoryLoader('./Dataset/', glob="./*.pdf", loader_cls=UnstructuredPDFLoader)

documents = loader.load()



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
chunk_vectors = [embeddings.embed(chunk) for chunk in chunks]





