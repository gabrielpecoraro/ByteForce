from langchain.embeddings import HuggingFaceEmbeddings
import os
from langchain.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


faiss_index_dir = "faiss_index"


print("queries")



if os.path.exists(faiss_index_dir):
    # Load the existing FAISS index (embeddings generation is skipped)
    faiss_index = FAISS.load_local(
        faiss_index_dir, embeddings, allow_dangerous_deserialization=True
    )
    print("Loaded existing FAISS index from", faiss_index_dir)
else:
    print("Must have a faiss_index to proceed")


query = "What is question 1"
query_vector = embeddings.embed_query(query)
results = faiss_index.similarity_search_by_vector(
    query_vector, k=5
)  # `k` is the number of top matches
for result in results:
    print(result.page_content)  # This shows the most relevant text chunks


