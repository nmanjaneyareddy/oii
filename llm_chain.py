from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st

def setup_qa_chain(vectorstore):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
    token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

    llm = HuggingFaceHub(
        repo_id=repo_id,
        huggingfacehub_api_token=token,
        model_kwargs={"temperature": 0.1, "max_new_tokens": 300}
    )

    # 🚫 NO context instructions, just a concise prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Answer the question strictly based on the context below.
Respond concisely. Do not explain your reasoning or include any instructions.

Context:
{context}

Question:
{question}

Answer in 1–2 sentences:"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )
