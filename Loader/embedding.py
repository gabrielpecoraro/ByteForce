import os
import time
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from mistralai import Mistral
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document


class EmbeddingGenerator:
    def __init__(self, api_key=None, model_name="mistral"):
        self.model_name = model_name

        if "mistral" in model_name.lower():
            if api_key is None:
                raise ValueError("API key must be provided for Mistral models.")
            self.client = Mistral(api_key=api_key)
            self.model_type = "mistral"
        else:
            self.client = SentenceTransformer(model_name)
            self.model_type = "local"

    def get_langchain_embedding_model(self):
        """Return a LangChain-compatible embedding model (for FAISS and RAG use)."""
        if self.model_type == "mistral":
            return MistralLangChainEmbedding(self.client, self.model_name)
        else:
            return HuggingFaceEmbeddings(model_name=self.model_name)

    def generate_embeddings(self, chunks, delay=1, max_retries=3, batch_size=32):
        chunk_vectors = []

        if self.model_type == "mistral":
            for chunk in tqdm(chunks, desc="Generating Mistral Embeddings"):
                retries = 0
                while True:
                    try:
                        response = self.client.embeddings.create(
                            model=self.model_name,
                            inputs=chunk,
                        )
                        chunk_vectors.append(response.data[0].embedding)
                        break
                    except Exception as e:
                        if "429" in str(e) or "rate limit" in str(e).lower():
                            retries += 1
                            if retries > max_retries:
                                raise Exception(
                                    f"Max retries exceeded for chunk: {chunk[:30]}"
                                ) from e
                            time.sleep(delay)
                        else:
                            raise e

        elif self.model_type == "local":
            print("Generating embeddings...")
            for i in tqdm(
                range(0, len(chunks), batch_size), desc="Generating embeddings"
            ):
                batch = chunks[i : i + batch_size]
                embeddings = self.client.encode(batch)
                chunk_vectors.extend(embeddings)

        return chunk_vectors

    def save_embeddings_to_faiss(self, chunks, save_path, metadata_list=None):
        if os.path.exists(save_path):
            print(f"FAISS index already exists at {save_path}. Loading...")
            vectorstore = FAISS.load_local(
                folder_path=save_path,
                embeddings=self.get_langchain_embedding_model(),
                allow_dangerous_deserialization=True,
            )
            return vectorstore

        print("Creating new FAISS index...")

        if metadata_list is None:
            metadata_list = [{} for _ in chunks]
        print("1")
        documents = [
            Document(page_content=chunk, metadata=meta)
            for chunk, meta in zip(chunks, metadata_list)
        ]
        print("2")
        vectorstore = FAISS.from_documents(
            documents, embedding=self.get_langchain_embedding_model()
        )
        vectorstore.save_local(save_path)
        print(f"Saved FAISS index to {save_path}")

        return vectorstore


class MistralLangChainEmbedding(Embeddings):
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def embed_documents(self, texts):
        return [
            self.client.embeddings.create(model=self.model_name, inputs=text)
            .data[0]
            .embedding
            for text in texts
        ]

    def embed_query(self, text):
        return (
            self.client.embeddings.create(model=self.model_name, inputs=text)
            .data[0]
            .embedding
        )
