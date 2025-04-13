import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def setup_qa_chain(vectorstore):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
    huggingfacehub_api_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

    # 🔧 Adjusted to limit verbosity
    llm = HuggingFaceHub(
        repo_id=repo_id,
        huggingfacehub_api_token=huggingfacehub_api_token,
        model_kwargs={
            "temperature": 0.0,           # Deterministic
            "max_new_tokens": 100         # Short and crisp output
        }
    )

    # 📝 Concise prompt for short answers
    concise_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an assistant for the IGIDR Library.
Using only the context below, answer the question in one clear and concise sentence.

Context:
{context}

Question:
{question}

Short Answer:
"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": concise_prompt},
        return_source_documents=False
    )

    return qa_chain
