import os
import pickle
import fitz  # PyMuPDF
from langchain.schema import Document

def load_and_save_pdf(pdf_file, doc_type, folder_path):
    # Define the filename for the corresponding .pkl file based on doc_type
    pkl_file = f"{doc_type}.pkl"
    
    # Initialize a list for documents
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


def detect_document_type(pdf_name):
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

def process_dataset(dataset_path):
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
            doc_type = detect_document_type(pdf_file)
            if doc_type != "unknown":  # Skip unsupported document types
                print(f"Processing '{pdf_file}' as type '{doc_type}'...")
                load_and_save_pdf(pdf_file, doc_type, dataset_path)
            else:
                print(f"Skipping '{pdf_file}' (unsupported document type).")
        else:
            print(f"Skipping '{pdf_file}' (not a PDF).")

