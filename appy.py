import streamlit as st
from loaders import load_documents, split_documents
from vectorstore import create_vector_store, load_vector_store
from llm_chain import setup_qa_chain
import os

st.title("📚 RAG Chatbot (PDF + HTML)")
st.write("Ask any question based on the documents.")

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
st.title("📚 Ask me")
query = st.chat_input("Ask about IGIDR Library")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain({"query": query})
        st.chat_message("user").write(query)
        st.chat_message("assistant").write(result["result"])

#        with st.expander("📄 Source Chunks"):
 #           for doc in result["source_documents"]:
  #              st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
   #             st.markdown(doc.page_content[:500] + "...")
