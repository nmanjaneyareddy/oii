import streamlit as st
from loaders import load_documents, split_documents
from vectorstore import create_vector_store, load_vector_store
from llm_chain import setup_qa_chain
import os
import re

# 🚿 Clean up verbose model output
def clean_answer(answer: str) -> str:
    patterns_to_strip = [
        r"^answer\s*[:,]?\s*",
        r"^based on .*?[.:]?\s*",
        r"^according to .*?[.:]?\s*",
        r"^use the following .*?[.:]?\s*",
        r"^from the context[:,]?\s*",
        r"^as per .*?[.:]?\s*",
        r"^the context .*?[.:]?\s*",
        r"^here is the answer[:,]?\s*"
    ]
    for pattern in patterns_to_strip:
        answer = re.sub(pattern, "", answer, flags=re.IGNORECASE)

    # Remove extra newlines and whitespace
    lines = answer.strip().splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    cleaned = " ".join(lines)

    # Limit to 2 sentences
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return " ".join(sentences[:2]).strip()

# 🧱 Streamlit setup
st.set_page_config(page_title="📚 IGIDRLIB Chatbot", page_icon="🤖")
st.title("🤖 IGIDRLIB Chatbot")
st.markdown("Ask any question related to the IGIDR Library.")

# 📥 Load or create FAISS vectorstore
if not os.path.exists("faiss_index"):
    with st.spinner("🔄 Processing documents..."):
        docs = load_documents()
        chunks = split_documents(docs)
        vectorstore = create_vector_store(chunks)
else:
    vectorstore = load_vector_store()

# 🧠 Load RetrievalQA chain
qa_chain = setup_qa_chain(vectorstore)

# 💬 Chat history session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🗨️ Chat input
user_input = st.chat_input("Ask about IGIDR Library...")

if user_input:
    with st.spinner("🤖 Thinking..."):
        result = qa_chain({"query": user_input})
        answer = clean_answer(result["result"])

        # Save question & answer
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", answer))

# 📜 Display chat history
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(f"🤖 {msg}")
