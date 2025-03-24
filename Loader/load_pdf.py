import os
import pickle
import fitz  # PyMuPDF
from langchain.schema import Document
import re
from termcolor import colored
from typing import List, Dict


class PDFLoader:
    def __init__(self):
        # For EPC
        self.EPCchapter_titles = ["Chapter", "Chapitre", "Kapitel"]
        self.EPCarticle_titles = ["Article", "Artikel", "Article"]

        # For PCT
        self.PCTarticle_titles = ["Article"]
        self.PCTrule_titles = ["Rule"]

        # For guidelines
        self.guidlinechapter_titles = ["Chapter"]

        # For exam
        self.examquestions_titles = ["Question"]

    def detect_document_type(self, pdf_name):
        """
        Detect the document type based on the name of the PDF file.
        """
        pdf_name_lower = pdf_name.lower()
        if "guidelines" in pdf_name_lower:
            return "guidelines"
        elif "pct" in pdf_name_lower:
            return "pct"
        elif "epc" in pdf_name_lower:
            return "epc"
        elif ("case_law" or "case law") in pdf_name_lower:
            return "case_law"
        elif ("answers" or "solution" or "solutions" or "answer") in pdf_name_lower:
            return "answers"  # For unsupported types
        else:
            return "questions"

    def detect_document_year(self, pdf_name):
        """
        Detect the year from the PDF file name. Assumes year format is 19xx or 20xx.
        """
        match = re.search(r'(19|20)\d{2}', pdf_name)
        if match:
            return int(match.group(0))
        else:
            raise ValueError(f"Year not found in '{pdf_name}'. Ensure filename contains a year (19xx or 20xx).")

    def load_and_save_pdf(self, pdf_file, doc_type, folder_path):
        """
        Load a PDF file and save extracted content into a .pkl file.
        If the file has already been processed (based on metadata "file_name"),
        it skips reprocessing.
        """
        pkl_folder = os.path.join(folder_path, "pkl")
        os.makedirs(pkl_folder, exist_ok=True)
        pkl_file = os.path.join(pkl_folder, f"{doc_type}.pkl")
        documents = []

        # Load existing documents if the pkl exists and check for duplicates
        if os.path.exists(pkl_file):
            with open(pkl_file, "rb") as f:
                existing_documents = pickle.load(f)
            # Check if this PDF file was already processed
            if any(doc.metadata.get("file_name") == pdf_file for doc in existing_documents):
                print(f"✔ '{pdf_file}' already exists in '{pkl_file}'. Skipping processing.")
                return existing_documents
            documents = existing_documents
        else:
            # Create an empty .pkl file if it doesn't exist
            with open(pkl_file, "wb") as f:
                pickle.dump(documents, f)
            print(f"✔ Created new '{pkl_file}' file.")

        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, pdf_file)
            try:
                pdf_document = fitz.open(pdf_path)
                print(f"✔ Successfully opened '{pdf_file}'. It has {pdf_document.page_count} pages.")

                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    text = page.get_text()
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"file_name": pdf_file, "page": page_num + 1},
                        )
                    )
                pdf_document.close()

                with open(pkl_file, "wb") as f:
                    pickle.dump(documents, f)
                    print(f"✔ Successfully updated '{pkl_file}' with new documents.")

            except Exception as e:
                print(f"⚠ Error reading '{pdf_file}': {e}")
        else:
            print(f"⚠ '{pdf_file}' is not a valid PDF file.")

        return documents

    def partition_and_structure_from_pkl(self, pkl_path, pdf_name):
        """
        Partition the data from a .pkl file into structured content with metadata:
        - document type
        - document year
        - chapters/articles/questions
        """
        doc_type = self.detect_document_type(pdf_name)
        doc_year = self.detect_document_year(pdf_name)

        try:
            with open(pkl_path, "rb") as f:
                documents = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{pkl_path}' does not exist.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading '{pkl_path}': {e}")

        structured_content = []

        if doc_type == "epc":
            chapters = []
            current_chapter = []
            for doc in documents:
                lines = doc.page_content.splitlines()
                for line_text in lines:
                    if any(title in line_text for title in self.EPCchapter_titles):
                        if current_chapter:
                            chapters.append(current_chapter)
                            current_chapter = []
                    current_chapter.append(line_text)
                if current_chapter:
                    chapters.append(current_chapter)

            chapters_with_articles = []
            for chapter in chapters:
                articles = []
                current_article = []
                for line in chapter:
                    if any(title in line for title in self.EPCarticle_titles):
                        if current_article:
                            articles.append("\n".join(current_article))
                            current_article = []
                    current_article.append(line)
                if current_article:
                    articles.append("\n".join(current_article))
                chapters_with_articles.append(articles)

            structured_content = chapters_with_articles

        elif doc_type == "pct":
            articles = []
            current_article = []
            for doc in documents:
                lines = doc.page_content.splitlines()
                for line_text in lines:
                    if any(title in line_text for title in self.PCTarticle_titles + self.PCTrule_titles) and self.is_bold(line_text):
                        if current_article:
                            articles.append("\n".join(current_article))
                            current_article = []
                    current_article.append(line_text)
                if current_article:
                    articles.append("\n".join(current_article))
            structured_content = articles
            

        elif doc_type == "guidelines":
            chapters = []
            current_chapter = []
            for doc in documents:
                lines = doc.page_content.splitlines()
                for line_text in lines:
                    if self.is_bold(line_text) and (
                        any(title in line_text for title in self.guidlinechapter_titles)
                        or re.match(r"^\d", line_text)
                        or re.match(r"^Chapter\s+[IVXLCDM]+", line_text)
                    ):
                        if current_chapter:
                            chapters.append(current_chapter)
                            current_chapter = []
                    current_chapter.append(line_text)
                if current_chapter:
                    chapters.append(current_chapter)
            structured_content = chapters

        elif doc_type == "case_law":
            chapters = []
            current_chapter = []
            for doc in documents:
                lines = doc.page_content.splitlines()
                for line_text in lines:
                    if re.match(r"^\d+(\.\d+)*\s", line_text) or re.match(r"^[a-zA-Z]\)\s", line_text):
                        if current_chapter:
                            chapters.append(current_chapter)
                            current_chapter = []
                    current_chapter.append(line_text)
                if current_chapter:
                    chapters.append(current_chapter)
            structured_content = chapters

        elif doc_type in ["questions", "answers"]:
            content_list = []
            current_item = []
            item_type = "question" if doc_type == "questions" else "answer"

            for doc in documents:
                lines = doc.page_content.splitlines()
                for line_text in lines:
                    is_new_item = (
                        (any(line_text.strip().startswith(title) for title in self.examquestions_titles)
                         or re.match(r"^\d+\.", line_text.strip()))
                        and self.is_bold(line_text)
                    )

                    if is_new_item:
                        if current_item:
                            content_list.append("\n".join(current_item).strip())
                            current_item = []
                    current_item.append(line_text)

                if current_item:
                    content_list.append("\n".join(current_item).strip())

            structured_content = {f"{item_type}s": content_list}

        return {
            "type": doc_type,
            "year": doc_year,
            "content": structured_content
        }

    def chunk_text(self, structured_dict, chunk_size=1024, overlap=256):
        """
        Chunk the structured content into pieces of roughly chunk_size words with an overlap.
        First, the content is flattened into a single string.
        """
        content = structured_dict["content"]
        # If content is a list (or nested lists), flatten it into one continuous string.
        if isinstance(content, list):
            flattened = []
            for item in content:
                if isinstance(item, list):
                    # Join inner lists with spaces
                    flattened.append(" ".join(item))
                else:
                    flattened.append(item)
            text = " ".join(flattened)
        else:
            text = content

        # Debug: print the total word count after flattening
        total_words = len(text.split())
        print("Total words after flattening:", total_words)

        words = text.split()
        chunks = [
            " ".join(words[i:i + chunk_size])
            for i in range(0, len(words), chunk_size - overlap)
        ]
        metadata_chunks = [
            {"type": structured_dict["type"], "year": structured_dict["year"], "chunk_index": idx}
            for idx, _ in enumerate(chunks)
        ]
        return [{"content": chunk, "metadata": metadata_chunks[idx]} for idx, chunk in enumerate(chunks)]

    def load_dataset(self, folder_path, chunk_size=1024, overlap=256) -> List[Dict]:
        """
        Process all PDF files in folder_path, extract, structure, and chunk their content.
        """
        pdf_files = [file for file in os.listdir(folder_path) if file.lower().endswith('.pdf')]
        all_chunks = []

        for pdf_file in pdf_files:
            print(colored(f"Processing '{pdf_file}'", "cyan"))
            doc_type = self.detect_document_type(pdf_file)
            self.load_and_save_pdf(pdf_file, doc_type, folder_path)
            pkl_path = os.path.join(folder_path, "pkl", f"{doc_type}.pkl")

            try:
                structured_dict = self.partition_and_structure_from_pkl(pkl_path, pdf_file)
                chunks = self.chunk_text(structured_dict, chunk_size, overlap)
                all_chunks.extend(chunks)
                print(colored(f"Successfully processed '{pdf_file}'", "green"))
            except Exception as e:
                print(colored(f"Failed to process '{pdf_file}': {e}", "red"))

        return all_chunks

    def is_bold(self, text):
        # Dummy implementation; replace with your actual logic for bold detection.
        return text.isupper()
