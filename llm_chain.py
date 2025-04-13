import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def setup_qa_chain(vectorstore):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
    token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

    llm = HuggingFaceHub(
        repo_id=repo_id,
        huggingfacehub_api_token=token,
        model_kwargs={"temperature": 0.2, "max_new_tokens": 512}
    )

    prompt = PromptTemplate.from_template("""
Answer the question based on the context below.

Context:
{context}

Question:
{question}

Answer:
""")

    parser = StrOutputParser()

    # Combine components using LCEL
    chain = prompt | llm | parser

    # Wrap the chain inside RetrievalQA-style interface
    def qa_chain(input_dict):
        retriever = vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(input_dict["query"])
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"result": chain.invoke({"context": context, "question": input_dict["query"]})}

    return qa_chain
