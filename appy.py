import streamlit as st
from loaders import load_documents, split_documents
from vectorstore import create_vector_store, load_vector_store
from llm_chain import setup_qa_chain
import os
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

st.set_page_config(page_title="IGIDR Library Chatbot", page_icon="📚")
st.title("📚 IGIDRLIB Chatbot")
st.write("Ask any question about IGIDR Library")

# Step 1: Load or build vectorstore
if not os.path.exists("faiss_index"):
    with st.spinner("Processing documents..."):
        docs = load_documents()
        chunks = split_documents(docs)
        vectorstore = create_vector_store(chunks)
else:
    vectorstore = load_vector_store()

# Step 2: Load LLM QA chain with concise mode
qa_chain = setup_qa_chain(vectorstore, concise=True)  # Assume concise=True triggers a short-answer prompt in your chain

# Step 3: Chat interface
query = st.text_input("Ask about IGIDR Library")

if query:
    with st.spinner("Generating answer..."):
        result = qa_chain({"query": query})
        answer = result["result"].strip()

        # Remove verbose prefaces
        prefixes_to_strip = [
            "Based on the context",
            "According to the documents",
            "Use the following pieces of context",
            "Here's what I found"
        ]
        for prefix in prefixes_to_strip:
            if answer.lower().startswith(prefix.lower()):
                answer = answer.split(":", 1)[-1].strip()

        # Keep only the first sentence (optional)
        if "." in answer:
            answer = answer.split(".")[0].strip() + "."

        st.markdown("### ❓ **Question**")
        st.write(query)
        st.markdown("### 🤖 **Answer**")
        st.write(answer)
