# vectorstore.py
# vectorstore_utils.py
def get_vectorstore_from_embeddings(embeddings, persist_directory=None):
    """
    Return an initialized vectorstore. Prefer Faiss if available,
    otherwise fall back to Chroma (works well on Streamlit Cloud).
    embeddings: LangChain/HF embeddings object
    persist_directory: optional directory for persistent store
    """
    try:
        import faiss  # noqa: F401
        from langchain.vectorstores import FAISS
        # Use FAISS (if you control runtime and have faiss installed)
        # Example: FAISS.from_documents(docs, embeddings)
        return "faiss"  # placeholder — replace with your FAISS setup
    except Exception:
        # Fallback to Chroma
        from langchain.vectorstores import Chroma
        from langchain.document_loaders import TextLoader  # example
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
