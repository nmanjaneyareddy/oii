# loaders.py
import os
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents():
    docs = []

    if os.path.exists("data/igidr_library_details.pdf"):
        pdf_loader = PyPDFLoader("data/igidr_library_details.pdf")
        docs += pdf_loader.load()
    else:
        print("⚠️ PDF file not found: data/igidr_library_details.pdf")

    if os.path.exists("data/library-2/"):
        html_loader = BSHTMLLoader("data/li.html")
        docs += html_loader.load()
    else:
        print("⚠️ HTML file not found: data/li.html")

    if not docs:
        raise ValueError("No valid documents found in the data/ directory.")

    return docs
