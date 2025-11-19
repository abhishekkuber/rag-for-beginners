from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

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

