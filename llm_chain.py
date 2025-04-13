import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def setup_qa_chain(vectorstore):
    # 🧠 Choose a Hugging Face model
    repo_id = "mistralai/Mistral-7B-Instruct-v0.1"  # You can replace this with another repo_id
    token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]  # Read token from Streamlit secrets

    # Initialize the LLM
    llm = HuggingFaceHub(
        repo_id=repo_id,
        huggingfacehub_api_token=token,
        model_kwargs={"temperature": 0.1, "max_new_tokens": 300}
    )

    # 🧾 Custom prompt to control verbosity and tone
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Answer the question strictly based on the context below.
Respond concisely. Do not explain your reasoning or include instructions.

Context:
{context}

Question:
{question}

Answer in 1–2 sentences:
"""
    )

    # Set up RetrievalQA chain with custom prompt
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )
