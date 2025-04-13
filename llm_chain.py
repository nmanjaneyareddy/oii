from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os

def setup_qa_chain(vectorstore):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
    huggingfacehub_api_token = os.environ.get(secret_key)

    if not huggingfacehub_api_token:
        raise ValueError("Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")

    llm = HuggingFaceHub(
        repo_id=repo_id,
        huggingfacehub_api_token=huggingfacehub_api_token,
        model_kwargs={"temperature": 0.2, "max_new_tokens": 512}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa_chain
