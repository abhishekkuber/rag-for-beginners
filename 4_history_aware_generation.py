# Code which retrieves the relevant documents from the vector database 
# and then uses a LLM to answer the question.
# Here, the can ask follow up questions, and the LLM can answer them while keeping the previous context in mind.

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv



def ask_question(query, chat_history, llm_model, retriever):
    print(f"ASKING QUESTION : {query}")
    if chat_history:
        print(f"Rewriting question based on the previous chat history...")
        messages = [SystemMessage(content="Given the chat history, only rewrite the question to be standnalone and searchable in a RAG, do not answer it. Return ONLY the rewritten question, NOTHING ELSE.")] + chat_history + [HumanMessage(content=query)]
        print(messages)
        input()
        response = llm_model.invoke(messages)
        rewritten_question = response.content.strip()
        print(f"\nBased on the history\n{query} is rewritten as \n{rewritten_question}!")
        query = rewritten_question
    
    relevant_docs = retriever.invoke(query)
    response = ask_standalone_question(query, relevant_docs, llm_model, chat_history)
    answer = response.content

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer : {answer}")
    return chat_history


def ask_standalone_question(query, relevant_docs, llm_model, chat_history):
    combined_input = f"""Based on the following documents and the chat history, please answer this question : {query}

    Documents : 
    {chr(10).join([f" - {doc.page_content}" for doc in relevant_docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer the question".
    """

    messages = [SystemMessage(content="You are a helpful assistant that answers questions based on the provided documents and chat history.")] + chat_history + [HumanMessage(content=combined_input)]
    response = llm_model.invoke(messages)
    return response


def start_chat(llm_model, retriever):
    print("Ask me questions! Type 'quit' if you want to exit.")
    chat_history = []
    while True:
        question = input("\nWhat do you want to ask?\n")
        if question.lower() == "quit":
            print("Goodbye, exiting the chat...")
            break
        
        chat_history = ask_question(question, chat_history, llm_model, retriever)


def main():
    load_dotenv()
    persist_directory = "db/chroma_db"

    # Load the embedding model
    model = HuggingFaceEmbeddings(model="intfloat/e5-large-v2")
    print("Loaded embedding model : intfloat/e5-large-v2!")

    # Connect to the vector database
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"Connected to the vector database at : {persist_directory}!")

    # Load the LLM model
    llm = HuggingFaceEndpoint(
        model="meta-llama/Llama-3.1-8B-Instruct", # you have to make sure that this model has an InferenceProvider on the HuggingFace Website.
        task="text-generation",
        max_new_tokens=200,
        temperature=0.7,
        provider="auto"
    )
    model = ChatHuggingFace(llm=llm)
    print("Loaded LLM model : meta-llama/Llama-3.1-8B-Instruct!")

    # Load the retriver
    retriever = db.as_retriever(search_kwargs={"k": 3}) # get the top 3 chunks

    start_chat(model, retriever)


if __name__ == "__main__":
    main()