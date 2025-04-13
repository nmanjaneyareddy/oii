from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import streamlit as st

def setup_qa_chain(vectorstore):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
    token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

    llm = HuggingFaceHub(
        repo_id=repo_id,
        huggingfacehub_api_token=token,
        model_kwargs={"temperature": 0.2, "max_new_tokens": 512}
    )

    # ✅ Clean, strict prompt – no fluff allowed
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Only answer the question using the context below.
If a relevant link is mentioned, include it.

Question: {question}
Context: {context}
Answer in 2-3 sentences:
"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )
