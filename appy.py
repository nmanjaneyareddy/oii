# app.py
import streamlit as st
from loaders import load_documents, split_documents
from vectorstore import create_vector_store, load_vector_store
from llm_chain import setup_qa_chain
import os

st.title("📚 RAG Chatbot with PDF + HTML")
st.write("Ask anything based on the uploaded documents.")

# Load or Create vectorstore
if not os.path.exists("faiss_index"):
    with st.spinner("Loading and processing documents..."):
        docs = load_documents()
        chunks = split_documents(docs)
        vectorstore = create_vector_store(chunks)
else:
    vectorstore = load_vector_store()

# Set up QA chain
qa_chain = setup_qa_chain(vectorstore)

# Chat interface
query = st.text_input("Enter your question:")
if query:
    with st.spinner("Thinking..."):
        result = qa_chain({"query": query})
        st.write("### 📌 Answer:")
        st.write(result["result"])

        with st.expander("📄 Sources"):
            for doc in result["source_documents"]:
                st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                st.markdown(doc.page_content[:300] + "...")
