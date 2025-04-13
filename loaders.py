from langchain.document_loaders import PyPDFLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents():
    pdf_loader = PyPDFLoader("data/example.pdf")
    html_loader = BSHTMLLoader("data/example.html")
    docs = pdf_loader.load() + html_loader.load()
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)
