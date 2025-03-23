import os
import pickle
import fitz  # PyMuPDF
from langchain.schema import Document
import re
from termcolor import colored


class PDFLoader:
    def __init__(self):
        # For EPC
        self.EPCchapter_titles = ["Chapter", "Chapitre", "Kapitel"]
        self.EPCarticle_titles = ["Article", "Artikel", "Article"]

        # For PCT
        self.PCTarticle_titles = ["Article"]
        self.PCTrule_titles = ["Rule"]

        # For guidlines
        self.guidlinechapter_titles = ["Chapter"]

        # For exam

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
        else:
            return "exam"  # For unsupported types

    def load_and_save_pdf(self, pdf_file, doc_type, folder_path):
        """
        Load a PDF file and save extracted content into a .pkl file.
        """
        pkl_folder = os.path.join(folder_path, "pkl")
        os.makedirs(
            pkl_folder, exist_ok=True
        )  # Create the pkl folder if it doesn't exist
        pkl_file = os.path.join(pkl_folder, f"{doc_type}.pkl")
        documents = []

        # Create an empty .pkl file if it doesn't exist
        if not os.path.exists(pkl_file):
            with open(pkl_file, "wb") as f:
                pickle.dump(documents, f)
            print(f"✔ Created new '{pkl_file}' file.")

        if pdf_file.endswith(".pdf"):  # Ensure it's a PDF
            pdf_path = os.path.join(folder_path, pdf_file)
            try:
                # Open the PDF
                pdf_document = fitz.open(pdf_path)
                print(
                    f"✔ Successfully opened '{pdf_file}'. It has {pdf_document.page_count} pages."
                )

                # Extract text from each page and store as a LangChain Document
                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    text = page.get_text()  # Extract text from the page
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"file_name": pdf_file, "page": page_num + 1},
                        )
                    )
                pdf_document.close()

                # Check if the .pkl file already exists
                if os.path.exists(pkl_file):
                    # Load existing data from the .pkl file
                    with open(pkl_file, "rb") as f:
                        existing_documents = pickle.load(f)
                    # Append new documents to the existing data
                    existing_documents.extend(documents)
                    documents = existing_documents

                # Save the updated list of documents into the .pkl file
                with open(pkl_file, "wb") as f:
                    pickle.dump(documents, f)
                    print(f"✔ Successfully updated '{pkl_file}' with new documents.")

            except Exception as e:
                print(f"⚠ Error reading '{pdf_file}': {e}")
        else:
            print(f"⚠ '{pdf_file}' is not a valid PDF file.")

        return documents

    def process_dataset(self, dataset_path):
        """
        Process all PDF files in the given dataset folder,
        detect their type, and save documents to corresponding .pkl files.
        """
        if not os.path.exists(dataset_path):
            print(f"⚠ The dataset path '{dataset_path}' does not exist.")
            return

        # Loop through all files in the dataset folder
        for pdf_file in os.listdir(dataset_path):
            if pdf_file.endswith(".pdf"):
                # Detect the document type from the file name
                doc_type = self.detect_document_type(pdf_file)
                if doc_type != "unknown":  # Skip unsupported document types
                    print(f"Processing '{pdf_file}' as type '{doc_type}'...")
                    self.load_and_save_pdf(pdf_file, doc_type, dataset_path)
                else:
                    print(f"Skipping '{pdf_file}' (unsupported document type).")
            else:
                print(f"Skipping '{pdf_file}' (not a PDF).")

    def is_bold(self, word):
        """
        Check if the word is bold.
        """
        return "fontname" in word and "Bold" in word["fontname"]

    def partition_in_chapter_and_article_from_pkl(self, pkl_path):
        """
        Partition the data from a .pkl file into chapters and articles within each chapter.
        """
        if type(pkl_path) != str:
            raise TypeError("pkl_path must be a string")

        chapters = []
        current_chapter = []

        # Load the data from the .pkl file
        try:
            with open(pkl_path, "rb") as f:
                documents = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{pkl_path}' does not exist.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading '{pkl_path}': {e}")

        # If file is EPC
        if "epc" in pkl_path:
            # Process each document (assuming each document has `page_content` containing text)
            for doc in documents:
                lines = (
                    doc.page_content.splitlines()
                )  # Split the page content into lines
                for line_text in lines:
                    # Check if the line matches chapter titles
                    if any(title in line_text for title in self.EPCchapter_titles):
                        if current_chapter:
                            chapters.append(current_chapter)
                            current_chapter = []
                    current_chapter.append(line_text)
                if current_chapter:  # Append the last chapter
                    chapters.append(current_chapter)

            # Partition chapters into articles
            chapters_with_articles = []
            for chapter in chapters:
                articles = []
                current_article = []
                for line in chapter:
                    # Check if the line matches article titles
                    if any(title in line for title in self.EPCarticle_titles):
                        if current_article:
                            articles.append("\n".join(current_article))
                            current_article = []
                    current_article.append(line)
                if current_article:  # Append the last article
                    articles.append("\n".join(current_article))
                chapters_with_articles.append(articles)

            return chapters_with_articles

        elif "pct" in pkl_path:
            chapters_with_articles = []
            current_article = []

            # Process each document (assuming each document has `page_content` containing text)
            for doc in documents:
                lines = (
                    doc.page_content.splitlines()
                )  # Split the page content into lines
                for line_text in lines:
                    # Check if the line matches article or rule titles in bold
                    if any(
                        title in line_text
                        for title in self.PCTarticle_titles + self.PCTrule_titles
                    ) and self.is_bold(line_text):
                        if current_article:
                            chapters_with_articles.append("\n".join(current_article))
                            current_article = []
                    current_article.append(line_text)
                if current_article:  # Append the last article
                    chapters_with_articles.append("\n".join(current_article))

            return chapters_with_articles

        elif "guidlines" in pkl_path:
            chapters = []
            current_chapter = []
            for doc in documents:
                lines = doc.page_content.splitlines()
                for line_text in lines:
                    # Check if the line is bold AND either:
                    # 1. Contains one of the guideline chapter titles, OR
                    # 2. Starts with a number
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
            # Depending on your needs, you could add an additional partitioning into articles here.
            # For now, we return the chapters.
            return chapters

        elif "exam" in pkl_path:
            paragraphs = []
            current_paragraph = []

            # Process each document (assuming each document has `page_content` containing text)
            for doc in documents:
                lines = (
                    doc.page_content.splitlines()
                )  # Split the page content into lines
                for line_text in lines:
                    # Check if the line is a full blank line
                    if line_text.strip() == "":
                        if current_paragraph:
                            paragraphs.append("\n".join(current_paragraph))
                            current_paragraph = []
                    else:
                        current_paragraph.append(line_text)
                if current_paragraph:  # Append the last paragraph
                    paragraphs.append("\n".join(current_paragraph))

            return paragraphs

    def chunk_text(self, text, chunk_size=512):
        """
        Chunk the text into smaller pieces.
        """
        words = text.split()
        chunks = [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]
        return chunks


if __name__ == "__main__":
    loader = PDFLoader()
    dataset_path = "../Dataset"
    pkl_folder = os.path.join(dataset_path, "pkl")
    os.makedirs(pkl_folder, exist_ok=True)  # Create the pkl folder if it doesn't exist

    # Process the dataset
    loader.process_dataset(dataset_path)

    # Assuming you want to process all .pkl files generated
    for pkl_file in os.listdir(pkl_folder):
        print(colored(f"Processing '{pkl_file}'...", "red"))
        if pkl_file.endswith(".pkl"):
            pkl_path = os.path.join(pkl_folder, pkl_file)
            parsed_content = loader.partition_in_chapter_and_article_from_pkl(pkl_path)

            # Print the chunked text for inspection
            if "exam" in pkl_file:
                for paragraph in parsed_content:
                    chunks = loader.chunk_text(paragraph)
                    print(
                        f"Paragraph: {paragraph[:50]}..."
                    )  # Print the first 50 characters of the paragraph for reference
                    print(f"Chunks: {chunks}")
                    print("-" * 80)
            else:
                for chapter in parsed_content:
                    for article in chapter:
                        chunks = loader.chunk_text(article)
                        print(
                            f"Article: {article[:50]}..."
                        )  # Print the first 50 characters of the article for reference
                        print(f"Chunks: {chunks}")
                        print("-" * 80)
