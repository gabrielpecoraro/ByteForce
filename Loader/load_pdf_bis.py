import os
import pickle
import fitz  # PyMuPDF
from langchain.schema import Document
from Loader.EPC import partition_in_chapter_and_article_from_pkl
from Loader.PCT import partition_in_chapter_rule_from_pkl


class PDFLoader_bis:
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
        elif "case law" in pdf_name_lower:
            return "case_law"
        else:
            return "unknown"  # For unsupported types

    def load_and_save_pdf(self, pdf_file, doc_type, folder_path):
        """
        Load a PDF file and save extracted content into a .pkl file.
        """
        pkl_file = f"{doc_type}.pkl"
        documents = []

        if pdf_file.endswith(".pdf"):  # Ensure it's a PDF
            pdf_path = os.path.join(folder_path, pdf_file)
            try:
                # Open the PDF
                pdf_document = fitz.open(pdf_path)
                print(f"✔ Successfully opened '{pdf_file}'. It has {pdf_document.page_count} pages.")

                # Extract text from each page and store as a LangChain Document
                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    text = page.get_text("text")  # Extract text from the page
                    fonts = page.get_text("dict")  # Extract font information for bold detection
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"file_name": pdf_file, "page": page_num + 1, "fonts": fonts},
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


    def split_content_by_type(self, pkl_path, doc_type, output_file=None):
        """
        Split the content of a .pkl file into chunks based on the document type.
        The returned value is a list of dictionaries with keys "content" and "heading".
        """
        try:
            with open(pkl_path, "rb") as f:
                documents = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{pkl_path}' does not exist.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading '{pkl_path}': {e}")

        content_dicts = []
        if doc_type == "epc":  # Use partitioning for 'epc'
            # partition_in_chapter_and_article_from_pkl returns a list of chapters,
            # each of which is a list of article strings.
            chapters_articles = partition_in_chapter_and_article_from_pkl(pkl_path)
            for chapter in chapters_articles:
                for article in chapter:
                    # Wrap each article in a dictionary.
                    content_dicts.append({"content": article, "heading": None})
        elif doc_type == "pct":  # Use partitioning for 'pct'
            # partition_in_chapter_rule_from_pkl returns a list of chapters (lists of lines)
            chapters = partition_in_chapter_rule_from_pkl(pkl_path)
            for chapter in chapters:
                # Join the lines to form a single text string.
                chapter_text = "\n".join(chapter)
                content_dicts.append({"content": chapter_text, "heading": None})
        else:
            # Fallback: combine all document content into one string.
            full_text = " ".join([doc.page_content for doc in documents])
            content_dicts.append({"content": full_text, "heading": None})

        # Optionally, save the dictionary list to a file.
        return content_dicts


    def chunk_text(self,text, chunk_size=512):
        """
        Chunk the text into smaller pieces.
        Handles lists of strings or single strings.
        """
        if isinstance(text, list):  # If text is already a list of strings
            words = " ".join(text).split()  # Combine the list into one string, then split into words
        else:  # If text is a single string
            words = text.split()
        
        # Create chunks of the specified size
        chunks = [
            " ".join(words[i: i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]
        return chunks
    
    def preprocess_and_save(self, pkl_file):
        """
        Preprocess text from a .pkl file and save it back with the same name.
        """
        try:
            # Load the existing .pkl file
            with open(pkl_file, "rb") as f:
                documents = pickle.load(f)

            # Example preprocessing (placeholder: modify as needed)
            for doc in documents:
                # Currently, it does nothing to the content. Customize preprocessing here.
                doc.metadata["processed"] = True  # Add a flag indicating preprocessing was done

            # Save the updated documents back to the same .pkl file
            with open(pkl_file, "wb") as f:
                pickle.dump(documents, f)
                print(f"✔ Successfully preprocessed and saved '{pkl_file}'.")

        except FileNotFoundError:
            print(f"⚠ The file '{pkl_file}' does not exist.")
        except Exception as e:
            print(f"⚠ An error occurred while processing '{pkl_file}': {e}")
 


    def load_dataset(self, dataset_path):
        """
        Load all PDF files in the dataset path, process them, and return all chunks with metadata.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"The dataset path '{dataset_path}' does not exist.")
        
        all_chunks = []

        # Iterate over all files in the dataset path
        for pdf_file in os.listdir(dataset_path):
            if pdf_file.endswith(".pdf"):  # Ensure the file is a PDF
                doc_type = self.detect_document_type(pdf_file)  # Detect document type
                folder_path = dataset_path

                # Step 1: Load and save the PDF into a .pkl file
                documents = self.load_and_save_pdf(pdf_file, doc_type, folder_path)
                
                
                # Step 2: Preprocess and save the .pkl file
                pkl_file = f"{doc_type}.pkl"
                #self.preprocess_and_save(pkl_file)

                # Step 3: Split the content based on document type
                split_output_file = f"{doc_type}_split.pkl"  # Optional: Save splits to a file
                split_chunks = self.split_content_by_type(pkl_file, doc_type, split_output_file)
                


                # Step 4: Perform chunking for each split
                for chunk in split_chunks:
                    chunked_content = self.chunk_text(chunk['content'])
                    for idx, text_chunk in enumerate(chunked_content):
                        chunk_metadata = {
                            "type": doc_type,
                            "file_name": pdf_file,
                            "heading": chunk.get('heading', None),
                            "chunk_index": idx + 1
                        }
                        all_chunks.append({"text": text_chunk, "metadata": chunk_metadata})

        
        # Return all chunks with metadata
        return all_chunks