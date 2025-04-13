import streamlit as st
from loaders import load_documents, split_documents
from vectorstore import create_vector_store, load_vector_store
from llm_chain import setup_qa_chain
import os
from utils import clean_answer
import re

st.set_page_config(page_title="📚 IGIDRLIB Chatbot", page_icon="🤖")
st.title("🤖 IGIDRLIB Chatbot")

# Load or build vectorstore
if not os.path.exists("faiss_index"):
    with st.spinner("Processing documents..."):
        docs = load_documents()
        chunks = split_documents(docs)
        vectorstore = create_vector_store(chunks)
else:
    vectorstore = load_vector_store()

# Load QA Chain
qa_chain = setup_qa_chain(vectorstore)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat-style input
user_input = st.chat_input("Ask me about IGIDR Library...")

if user_input:
    with st.spinner("🤖 Thinking..."):
        result = qa_chain({"query": user_input})
        answer = clean_answer(result["result"])

        # Optional: strip known verbose starts
        def clean_answer(answer: str) -> str:
            patterns_to_strip = [
            r"^based on the context[:,]?\s*",
            r"^according to (the )?document[s]?[,:]?\s*",
            r"^use the following pieces of context[:,]?\s*",
            r"^from the context[:,]?\s*",
            r"^as per the context[:,]?\s*",
                ]
            for pattern in patterns_to_strip:
            answer = re.sub(pattern, "", answer, flags=re.IGNORECASE)
            return answer.strip()

        # Save to chat history
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", answer))

# Render chat history
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)
