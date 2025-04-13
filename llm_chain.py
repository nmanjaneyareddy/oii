import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
def setup_qa_chain(vectorstore, concise=False):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
    huggingfacehub_api_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

    llm = HuggingFaceHub(
        repo_id=repo_id,
        huggingfacehub_api_token=huggingfacehub_api_token,
        model_kwargs={
            "temperature": 0.0,
            "max_new_tokens": 100 if concise else 512
        }
    )

    prompt_template = """
You are an assistant for the IGIDR Library.
Using only the context below, answer the question {style}

Context:
{context}

Question:
{question}

Answer:
"""

    style = "in one short and clear sentence" if concise else "as clearly and completely as possible"

    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template.format(style=style)
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=False
    )

    return qa_chain
