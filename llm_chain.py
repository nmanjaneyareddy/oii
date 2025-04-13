import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def setup_qa_chain(vectorstore):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

    # HuggingFace model setup
    llm = HuggingFaceHub(
        repo_id=repo_id,
        huggingfacehub_api_token=token,
        model_kwargs={"temperature": 0.1, "max_new_tokens": 300}
    )

    # Clean prompt — no verbose instructions
    prompt = PromptTemplate.from_template("""
Answer concisely based only on the context below.

Context:
{context}

Question:
{question}

Answer:
""")

    parser = StrOutputParser()

    # Chain together prompt → model → parser
    chain = prompt | model | parser

    # Wrap this into a callable function
    def qa_chain(input_dict):
        retriever = vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(input_dict["query"])
        context = "\n\n".join(doc.page_content for doc in docs)
        result = chain.invoke({"context": context, "question":  input_dict["query"]})
        return {"result": result.strip()}

    return qa_chain
