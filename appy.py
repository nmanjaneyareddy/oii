import streamlit as st
from loaders import load_documents, split_documents
from vectorstore import create_vector_store, load_vector_store
from llm_chain import setup_qa_chain
import os
import re

def clean_answer(text):
    # Remove instructions and labels like "Context:" and "Answer:"
    text = re.sub(r"(?i)(use the following.*?answer:)", "", text, flags=re.DOTALL)
    text = re.sub(r"(?i)^context:|^answer:", "", text)
    return text.strip()

answer = clean_answer(result.get("result", ""))

# 🎛️ Streamlit page setup
st.set_page_config(page_title="📚 IGIDRLib Chatbot", page_icon="")
st.title("🤖 IGIDRLIB Chatbot")
st.markdown("Ask anything about your Library.")

# 📦 Load or build vectorstore
if not os.path.exists("faiss_index"):
    with st.spinner("🔄 Processing documents..."):
        docs = load_documents()
        chunks = split_documents(docs)
        vectorstore = create_vector_store(chunks)
else:
    vectorstore = load_vector_store()

# 🤖 Setup QA chain
qa_chain = setup_qa_chain(vectorstore)

# 💬 Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 📩 User chat input
user_input = st.chat_input("Ask about IGIDR Library...")

if user_input:
    with st.spinner("🤖 Thinking..."):
        result = qa_chain({"query": user_input})
        answer = result.get("result", "").strip()

        # Save to session chat history
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", answer))

# 💬 Display chat history
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(f"🤖 {msg}")
