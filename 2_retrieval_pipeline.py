# Code which retrieves the relevant documents from the vector database 
# and then uses a LLM to answer the question.
# Please note that this is a one off generation (this means that this is not a conversational RAG)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persist_directory = "db/chroma_db"

model = HuggingFaceEmbeddings(model="intfloat/e5-large-v2")
print("Loaded model : intfloat/e5-large-v2!")

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=model,
    collection_metadata={"hnsw:space": "cosine"}
)

# search query to search relevant documents 

# query = "Which island does SpaceX lease for its launches in the Pacific?"
# query = "Has Tesla been involved in any lawsuits?"
# query = "What is the working philosophy at Google?"
query = "What was Nvidia's first product?"


retriever = db.as_retriever(search_kwargs={"k": 3}) # get the top 3 chunks

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3
#     }    
# )

relevant_docs = retriever.invoke(query)

# print(f"User Query : {query}")
# print("--- Context ---")
# for i, doc in enumerate(relevant_docs):
#     print(f"Document {i}\n{doc.page_content}")
#     print("-"*100)



############# Create a generation model (LLM) #############
combined_input = f"""Based on the following documents, please answer this question : {query}

Documents : 
{chr(10).join([f" - {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer the question".
"""

llm = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=200,
    temperature=0.7,
    provider="auto"
)

model = ChatHuggingFace(llm=llm)

# messages = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant which answers questions based on the provided documents."),
#     ("user", combined_input)
# ])

messages = [
    SystemMessage(content="You are a helpful assistant that answers questions based on the provided documents. If you can't find the answer in the "),
    HumanMessage(content=combined_input)
]

response = model.invoke(messages)

print(response.content)