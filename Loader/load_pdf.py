import os
import pickle
import fitz  # PyMuPDF
import pdfplumber
from langchain_core.documents import Document
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
        pkl_folder = os.path.join(folder_path, "pkl")
        os.makedirs(pkl_folder, exist_ok=True)
        pkl_file = os.path.join(pkl_folder, f"{doc_type}.pkl")
        documents = []

        # Load existing documents if pkl exists
        if os.path.exists(pkl_file):
            with open(pkl_file, "rb") as f:
                existing_documents = pickle.load(f)
            if any(doc.metadata.get("file_name") == pdf_file for doc in existing_documents):
                print(f"✔ '{pdf_file}' already exists in '{pkl_file}'. Skipping processing.")
                return existing_documents
            documents = existing_documents
        else:
            with open(pkl_file, "wb") as f:
                pickle.dump(documents, f)
            print(f"✔ Created new '{pkl_file}' file.")

        pdf_path = os.path.join(folder_path, pdf_file)

        if pdf_file.endswith(".pdf"):
            try:
                
                with pdfplumber.open(pdf_path) as pdf:
                    print(f"✔ Successfully opened '{pdf_file}'. It has {len(pdf.pages)} pages.")
                    for page_num, page in enumerate(pdf.pages):
                        # Extract text with layout mode enabled
                        text = page.extract_text(layout=True)
                        bolded_lines = []
                        if text:
                            lines = text.splitlines()
                            # Use page.chars to do a simple bold heuristic:
                            for line in lines:
                                # Check if any character in the page has 'Bold' in its fontname and is present in this line
                                is_bold = any("Bold" in char.get("fontname", "") for char in page.chars if char["text"].strip() in line)
                                bolded_lines.append((line.strip(), is_bold))
                        else:
                            text = ""
                        documents.append(
                            Document(
                                page_content=text.strip(),
                                metadata={
                                    "file_name": pdf_file,
                                    "page": page_num + 1,
                                    "bolded_text": bolded_lines
                                },
                            )
                        )
                with open(pkl_file, "wb") as f:
                    pickle.dump(documents, f)
                    print(f"✔ Successfully updated '{pkl_file}' with pdfplumber documents.")
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
                metadata = doc.metadata
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
                metadata = doc.metadata
                lines = doc.page_content.splitlines()
                for line_text in lines:
                    if any(title in line_text for title in self.PCTarticle_titles + self.PCTrule_titles) and self.is_bold(metadata, line_text):
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
                metadata = doc.metadata
                lines = doc.page_content.splitlines()
                for line_text in lines:
                    if self.is_bold(metadata, line_text) and (
                        any(title in line_text for title in self.guidlinechapter_titles)
                        or re.match(r"^\d", line_text)
                        or re.match(r"^Chapter\s+[IVXLCDM]+", line_text)
                    ):
                        if current_chapter:
                            chapters.append("\n".join(current_chapter))
                            current_chapter = []
                    current_chapter.append(line_text)
                if current_chapter:
                    chapters.append("\n".join(current_chapter))
            structured_content = chapters

        elif doc_type == "case_law":
            chapters = []
            current_chapter = []
            for doc in documents:
                metadata = doc.metadata
                lines = doc.page_content.splitlines()
                for line_text in lines:
                    if re.match(r"^\d+(\.\d+)*\s", line_text) or re.match(r"^[a-zA-Z]\)\s", line_text):
                        if current_chapter:
                            chapters.append("\n".join(current_chapter))
                            current_chapter = []
                    current_chapter.append(line_text)
                if current_chapter:
                    chapters.append("\n".join(current_chapter))
            structured_content = chapters

        elif doc_type in ["questions", "answers"]:
            content_list = []
            current_item = []
            item_type = "question" if doc_type == "questions" else "answer"

            for doc in documents:
                metadata = doc.metadata
                lines = doc.page_content.splitlines()
                for line_text in lines:
                    is_new_item = (
                        (any(line_text.strip().startswith(title) for title in self.examquestions_titles)
                         or re.match(r"^\d+\.", line_text.strip()))
                        and self.is_bold(metadata, line_text)
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

    def chunk_text_adapt(self, structured_dict, min_chunk_size=50, max_chunk_size=512, overlap=50):
        """
        Adaptively chunk structured content into meaningful pieces based on metadata (document type).
        This method ensures chunks are neither too small nor exceed a maximum length.
        """
        import re

        doc_type = structured_dict["type"]
        content = structured_dict["content"]

        chunks = []

        if isinstance(content, dict):
            content = content[next(iter(content))]

        if doc_type == "epc":
            sections = [article for chapter in content for article in chapter]

        elif doc_type in ["pct", "guidelines", "case_law"]:
            sections = [article for article in content]

        elif doc_type in ["questions", "answers"]:
            sections = [article for article in content]

        else:
            sections = [" ".join(item) if isinstance(item, list) else item for item in content]

        current_chunk = []
        current_length = 0

        def flush_chunk():
            nonlocal current_chunk, current_length
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

        for section in sections:
            sentences = re.split(r'(?<=[.!?]) +', section.strip())
            for sentence in sentences:
                sentence_length = len(sentence.split())

                if current_length + sentence_length <= max_chunk_size:
                    current_chunk.append(sentence)
                    current_length += sentence_length
                else:
                    if current_length >= min_chunk_size:
                        flush_chunk()
                        current_chunk.append(sentence)
                        current_length = sentence_length
                    else:
                        words = sentence.split()
                        for i in range(0, len(words), max_chunk_size - overlap):
                            chunk_slice = words[i:i + max_chunk_size]
                            chunks.append(" ".join(chunk_slice))
                        current_chunk = []
                        current_length = 0

        flush_chunk()

        metadata_chunks = [
            {"type": doc_type, "year": structured_dict["year"], "chunk_index": idx}
            for idx, _ in enumerate(chunks)
        ]

        return [
            {"content": chunk, "metadata": metadata_chunks[idx]}
            for idx, chunk in enumerate(chunks)
        ]

    def chunk_text(self, structured_dict, chunk_size=1024, overlap=256):
        """
        Chunk structured content into pieces based on metadata (document type).
        Each document type is processed differently for meaningful embeddings.
        """
        doc_type = structured_dict["type"]
        content = structured_dict["content"]

        chunks = []

        if doc_type == "epc":
            # EPC: Chunk by articles within chapters
            for chapter in content:
                for article in chapter:
                    words = article.split()
                    for i in range(0, len(words), chunk_size - overlap):
                        chunk = " ".join(words[i:i + chunk_size])
                        chunks.append(chunk)

        elif doc_type == "pct":
            # PCT: Chunk each article or rule separately
            for article in content:
                words = article.split()
                for i in range(0, len(words), chunk_size - overlap):
                    chunk = " ".join(words[i:i + chunk_size])
                    chunks.append(chunk)

        elif doc_type == "guidelines":
            # Guidelines: Chunk by chapter
            for chapter in content:
                words = chapter.split()
                for i in range(0, len(words), chunk_size - overlap):
                    chunk = " ".join(words[i:i + chunk_size])
                    chunks.append(chunk)

        elif doc_type == "case_law":
            # Case law: Chunk by identified chapters
            for chapter in content:
                words = chapter.split()
                for i in range(0, len(words), chunk_size - overlap):
                    chunk = " ".join(words[i:i + chunk_size])
                    chunks.append(chunk)

        elif doc_type in ["questions", "answers"]:
            # Questions/answers: chunk each question or answer separately
            items = content["questions"] if doc_type == "questions" else content["answers"]
            for item in items:
                words = item.split()
                for i in range(0, len(words), chunk_size - overlap):
                    chunk = " ".join(words[i:i + chunk_size])
                    chunks.append(chunk)

        else:
            # Fallback for any other document types
            flattened_text = " ".join([" ".join(item) if isinstance(item, list) else item for item in content])
            words = flattened_text.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                chunks.append(chunk)

        total_chunks = len(chunk)
        print("Total words after flattening:", total_chunks)

        # Attach metadata to each chunk
        metadata_chunks = [{"type": doc_type, "year": structured_dict["year"], "chunk_index": idx} for idx in range(len(chunks))]

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
                print(colored(f"Chunking '{pdf_file}'", "magenta"))
                chunks = self.chunk_text(structured_dict, chunk_size, overlap)
                #chunks = self.chunk_text_adapt(structured_dict, min_chunk_size=50, max_chunk_size=512, overlap=50)
                all_chunks.extend(chunks)
                print(colored(f"Successfully processed '{pdf_file}'", "green"))
            except Exception as e:
                print(colored(f"Failed to process '{pdf_file}': {e}", "red"))

        return all_chunks

    def is_bold(self, metadata, line_text):
        bolded_text = metadata.get("bolded_text", [])
        return any(line.strip() == line_text.strip() and bold for line, bold in bolded_text)


# if __name__ == "__main__":
#     import sys
#     from termcolor import colored

#     # Create an instance of your PDFLoader
#     loader = PDFLoader()

#     # 1. Parse command-line arguments (optional)
#     #    e.g., let the user pass the folder path as an argument:
#     if len(sys.argv) > 1:
#         folder_path = sys.argv[1]
#     else:
#         # Fallback: set a default folder or prompt the user
#         folder_path = "Dataset_Bis"  # Replace with your default folder path

#     print(colored(f"Debugging PDF loading from folder: {folder_path}", "cyan"))

#     # 2. Call the loading and dataset method
#     chunks = loader.load_dataset(folder_path)

#     # 3. Print out how many chunks we got
#     print(colored(f"Total chunks: {len(chunks)}", "green"))

#     # 4. (Optional) Look for specific content, e.g., "Article 52"
#     found_article_52 = False
#     for i, chunk_data in enumerate(chunks):
#         content = chunk_data["content"]
#         if "Article 52" in content or "Art. 52" in content:
#             found_article_52 = True
#             print(colored(f"\nFound 'Article 52' in chunk index {i}", "yellow"))
#             print(content)  # Show first 300 characters
#             # break  # you could break here if you only want the first match

#     if not found_article_52:
#         print(colored("No 'Article 52' text found in any chunk.", "red"))

#     print(colored("Debugging complete.", "cyan"))

