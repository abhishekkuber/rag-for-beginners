from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


persist_directory = "db/chroma_db"

model = HuggingFaceEmbeddings(model="intfloat/e5-large-v2")
print(f"Loaded embedding model!")

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=model,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "How much did Microsoft pay to acquire Github?"


'''
# METHOD 1 : Basic Similarity Search
# Returns the top k most similar documents
print(f"METHOD 1 : Similarity Search (k=3)")
retriever = db.as_retriever(searc_kwargs={"k": 3})
docs = retriever.invoke(query)
for i, doc in enumerate(docs):
    print(f"Document {i}\n{doc.page_content}\n------------------------------------------\n")
'''


'''
# METHOD 2 : Similarity with score threshold
# Returns the top k most similar documents  
print(f"METHOD 1 : Similarity Search (k=3)")
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.3
    }
)
docs = retriever.invoke(query)
for i, doc in enumerate(docs):
    print(f"Document {i}\n{doc.page_content}\n------------------------------------------\n")
'''


# METHOD 3 : Maximum Marginal Relevance (MMR)
print(f"METHOD 3 : MMR")
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)
docs = retriever.invoke(query)
for i, doc in enumerate(docs):
    print(f"Document {i}\n{doc.page_content}\n------------------------------------------\n")