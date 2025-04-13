import streamlit as st
from loaders import load_documents, split_documents
from vectorstore import create_vector_store, load_vector_store
from llm_chain import setup_qa_chain
import os

st.title("📚 IGIDRLIB Chatbot)")
st.write("Ask any question about IGIDR Library")

# Step 1: Load or build vectorstore
if not os.path.exists("faiss_index"):
    with st.spinner("Processing documents..."):
        docs = load_documents()
        chunks = split_documents(docs)
        vectorstore = create_vector_store(chunks)
else:
    vectorstore = load_vector_store()

# Step 2: Load LLM QA chain
qa_chain = setup_qa_chain(vectorstore)

# Step 3: Chat interface
query = st.text_input("Ask about IGIDR Library")

if query:
    with st.spinner("Generating answer..."):
        result = qa_chain({"query": query})
        
        st.markdown("### ❓ **Question**")
        st.write(query)
        
        st.markdown("### 🤖 **Answer**")
        st.write(result["result"])
