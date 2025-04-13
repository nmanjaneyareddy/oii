import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def setup_qa_chain(vectorstore):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
    token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

    # Initialize the language model from Hugging Face
    llm = HuggingFaceHub(
        repo_id=repo_id,
        huggingfacehub_api_token=token,
        model_kwargs={
            "temperature": 0.2,
            "max_new_tokens": 1000
        }
    )

    # A mild custom prompt that gives enough flexibility to the LLM
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use the following context to answer the user's question.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # Create a RetrievalQA chain with this prompt
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )
