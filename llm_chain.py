from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

def setup_qa_chain(vectorstore):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.1"  # or any open-access LLM
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.2, "max_new_tokens": 512})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa_chain
