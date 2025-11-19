# Code which reads the text documents, splits them into chunks, and stores them in a vector database using the HF intfloat/e5-large-v2 model. 

import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


def load_documents(document_path="docs/"):
    print(f"Loading documents from {document_path}...")
    
    loader = DirectoryLoader(
        document_path,
        glob="*.txt", # load only the text files
        loader_cls=TextLoader) # use the TextLoader to load the text files (since we are working with text files, there are more loader classes to choose from)
    
    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No documents found in the directory {document_path}.")
    
    print(f"Loaded {len(documents)} documents from {document_path}!")
    # for doc in documents:
    #     print(f"Content length: {len(doc.page_content)} characters")
    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=0):
    print("Splitting documents into chunks...")
    # CharacterTextSplitter is a text splitter that splits the documents into chunks of a certain size
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if len(chunks) == 0:
        raise ValueError(f"No chunks were generated from the documents.\nChunk size: {chunk_size}\nChunk overlap: {chunk_overlap}")    
    print(f"Success! Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks


def create_vector_store(chunks, save_dir="db/chroma_db"):
    print("Embedding the chunks and storing them in a vector database (ChromaDB)")
    
    # https://modal.com/blog/embedding-models-article
    model = HuggingFaceEmbeddings(model="intfloat/e5-large-v2")
    
    # model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create a ChromaDB vector store
    print("Creating a ChromaDB vector store....")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=model,
        persist_directory=save_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("Finished adding documents to the vector store.")
    return vectorstore

    


def main():
    print("Starting the ingestion pipeline...")
    
    # Step 1 : Loading the documents (Reading text files)
    documents = load_documents()

    # Step 2 : Splitting the documents into chunks. According to the course, it is probably the most important step, as it is the foundation of the RAG pipeline
    chunks = split_documents(documents)

    # Step 3 : Embed the chunks and store them in a vector database.
    vectorstore = create_vector_store(chunks)


if __name__ == "__main__":
    main()