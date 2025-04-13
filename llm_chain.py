import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def setup_qa_chain(vectorstore):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
    huggingfacehub_api_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

    llm = HuggingFaceHub(
        repo_id=repo_id,
        huggingfacehub_api_token=huggingfacehub_api_token,
        model_kwargs={"temperature": 0.2, "max_new_tokens": 512}
    )

    # ✅ Minimal clean prompt (no context instructions)
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Answer the following question based only on the provided context.

Question: {question}
Context: {context}
Answer:
"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=False  # ✅ disables source chunks
    )

    return qa_chain
