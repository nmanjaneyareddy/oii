# loaders.py
import os
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents():
    docs = []

    if os.path.exists("data/example.pdf"):
        pdf_loader = PyPDFLoader("data/example.pdf")
        docs += pdf_loader.load()
    else:
        print("⚠️ PDF file not found: data/example.pdf")

    if os.path.exists("data/example.html"):
        html_loader = BSHTMLLoader("data/example.html")
        docs += html_loader.load()
    else:
        print("⚠️ HTML file not found: data/example.html")

    if not docs:
        raise ValueError("No valid documents found in the data/ directory.")

    return docs
