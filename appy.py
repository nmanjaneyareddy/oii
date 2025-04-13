import streamlit as st
from loaders import load_documents, split_documents
from vectorstore import create_vector_store, load_vector_store
from llm_chain import setup_qa_chain
import os
import re

# 🧼 Clean up verbose LLM output
def clean_answer(answer: str) -> str:
    patterns_to_strip = [
        r"^use the following.*?\n",    # Removes "Use the following context..."
        r"^context:.*?\n",             # Removes "Context: ..."
        r"^answer[:,]?\s*",            # Removes "Answer:"
    ]
    for pattern in patterns_to_strip:
        answer = re.sub(pattern, "", answer, flags=re.IGNORECASE | re.DOTALL)
    return answer.strip()

# 🎛️ Streamlit page setup
st.set_page_config(page_title="📚 IGIDRLIB Chatbot", page_icon="🤖")
st.title("🤖 IGIDRLIB Chatbot")
st.markdown("Ask any question related to the IGIDR Library.")

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
        raw_answer = result.get("result", "").strip()
        answer = clean_answer(raw_answer)

        # ✅ Save question and cleaned answer to chat history
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", answer))

# 💬 Display chat history (user + bot)
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(f"❓ {msg}")
    else:
        st.chat_message("assistant").write(f"🤖 {msg}")
