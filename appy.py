import streamlit as st
from loaders import load_documents, split_documents
from vectorstore import create_vector_store, load_vector_store
from llm_chain import setup_qa_chain
import os

st.set_page_config(page_title="📚 IGIDRLIB Chatbot", page_icon="🤖")
st.title("🤖 IGIDRLIB Chatbot")
st.markdown("Ask any question about IGIDR Library.")

# Load or build vectorstore
if not os.path.exists("faiss_index"):
    with st.spinner("🔄 Processing documents..."):
        docs = load_documents()
        chunks = split_documents(docs)
        vectorstore = create_vector_store(chunks)
else:
    vectorstore = load_vector_store()

qa_chain = setup_qa_chain(vectorstore)

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.chat_input("Ask about IGIDR Library...")

if user_input:
    with st.spinner("🤖 Thinking..."):
        result = qa_chain({"query": user_input})
        answer = result.get("result", "")

        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", answer))

# Display history
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(f"❓ {msg}")
    else:
        st.chat_message("assistant").write(f"🤖 {msg}")
