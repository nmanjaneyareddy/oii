from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM  # or use HuggingFaceHub if needed

def setup_qa_chain(vectorstore):
    # ✅ Use Ollama locally (replace with HuggingFaceHub if needed)
    llm = OllamaLLM(model="mistral")  # you can replace this with `HuggingFaceHub(...)`

    # ✅ Chat-style prompt — cleaner and natural
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant answering questions about a library. Be brief and to the point."),
        ("user", "Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}")
    ])

    parser = StrOutputParser()
    chain = prompt | llm | parser

    # Wrapper to integrate with your app
    def qa_chain(input_dict):
        retriever = vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(input_dict["query"])
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"result": chain.invoke({"context": context, "question": input_dict["query"]}).strip()}

    return qa_chain
