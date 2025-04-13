import streamlit as st
from loaders import load_documents, split_documents
from vectorstore import create_vector_store, load_vector_store
from llm_chain import setup_qa_chain
import os

st.set_page_config(page_title="📚 IGIDRLIB Chatbot", page_icon="🤖")
st.title("🤖 IGIDRLIB Chatbot")
st.markdown("Ask any question related to the IGIDR Library.")

# Step 1: Load or create vector store
if not os.path.exists("faiss_index"):
    with st.spinner("Processing documents..."):
        docs = load_documents()
        chunks = split_documents(docs)
        vectorstore = create_vector_store(chunks)
else:
    vectorstore = load_vector_store()

# Step 2: Load QA chain
qa_chain = setup_qa_chain(vectorstore)

# Step 3: Session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Step 4: Chat input
user_input = st.chat_input("Ask about IGIDR Library...")

if user_input:
    with st.spinner("🤖 Generating answer..."):
        result = qa_chain({"query": user_input})
        answer = result["result"].strip()

        # Optional: clean verbose responses
        for prefix in [
            "Based on the context",
            "According to the documents",
            "Use the following"
        ]:
            if answer.lower().startswith(prefix.lower()):
                answer = answer.split(":", 1)[-1].strip()

        # Save conversation
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", answer))

# Step 5: Display chat history
for role, message in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(f"🤖 {message}")
