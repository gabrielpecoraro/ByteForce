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
import gensim
import gensim.corpora as corpora
from nltk.corpus import stopwords


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

print("splitting pdfs...")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

chunks = text_splitter.split_documents(documents)

print("topic modeling ")


# LDA Topic Modeling
def preprocess_text(text):
    tokens = text.lower().split()
    tokens = [
        word.strip(string.punctuation)
        for word in tokens
        if word.strip(string.punctuation)
    ]
    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word not in stop_words]


docs = [chunk.page_content for chunk in chunks if chunk.page_content.strip()]
processed_docs = [preprocess_text(doc) for doc in docs]
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(text) for text in processed_docs]

lda_model = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

for i, tokens in enumerate(processed_docs):
    bow = dictionary.doc2bow(tokens)
    topic_probs = lda_model.get_document_topics(bow)
    dominant_topic = max(topic_probs, key=lambda x: x[1])[0] if topic_probs else None
    chunks[i].metadata["dominant_topic"] = dominant_topic


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
