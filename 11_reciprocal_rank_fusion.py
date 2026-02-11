from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers.fix import OutputFixingParser
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from collections import defaultdict


load_dotenv()

def reciprocal_rank_fusion(extracted_chunks, k=60):
    scores = {}
    for query_chunks in extracted_chunks:
        for i, doc in enumerate(query_chunks):
            if not doc.id in scores:
                scores[doc.id] = [doc, 1 / (k+i+1)]
            else:
                prev_score = scores[doc.id][-1]
                prev_score += (1/(k+i+1))
                scores[doc.id][-1] = prev_score
    
    sorted_chunks = dict(sorted(scores.items(), key=lambda item: item[1][1], reverse=True))
    return sorted_chunks

'''
# HARISH NEEL'S FUNCTION
def rrf(chunk_lists, k=60, verbose=True):

    if verbose:
        print("\n" + "="*60)
        print("APPLYING RECIPROCAL RANK FUSION")
        print("="*60)
        print(f"\nUsing k={k}")
        print("Calculating RRF scores...\n")
    
    # Data structures for RRF calculation
    rrf_scores = defaultdict(float)  # Will store: {chunk_content: rrf_score}
    all_unique_chunks = {}  # Will store: {chunk_content: actual_chunk_object}
    
    # For verbose output - track chunk IDs
    chunk_id_map = {}
    chunk_counter = 1
    
    # Go through each retrieval result
    for query_idx, chunks in enumerate(chunk_lists, 1):
        if verbose:
            print(f"Processing Query {query_idx} results:")
        
        # Go through each chunk in this query's results
        for position, chunk in enumerate(chunks, 1):  # position is 1-indexed
            # Use chunk content as unique identifier
            chunk_content = chunk.page_content
            
            # Assign a simple ID if we haven't seen this chunk before
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1
            
            chunk_id = chunk_id_map[chunk_content]
            
            # Store the chunk object (in case we haven't seen it before)
            all_unique_chunks[chunk_content] = chunk
            
            # Calculate position score: 1/(k + position)
            position_score = 1 / (k + position)
            
            # Add to RRF score
            rrf_scores[chunk_content] += position_score
            
            if verbose:
                print(f"  Position {position}: {chunk_id} +{position_score:.4f} (running total: {rrf_scores[chunk_content]:.4f})")
                print(f"    Preview: {chunk_content[:80]}...")
        
        if verbose:
            print()
    
    # Sort chunks by RRF score (highest first)
    sorted_chunks = sorted(
        [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()],
        key=lambda x: x[1],  # Sort by RRF score
        reverse=True  # Highest scores first
    )
    
    if verbose:
        print(f"✅ RRF Complete! Processed {len(sorted_chunks)} unique chunks from {len(chunk_lists)} queries.")
    
    return sorted_chunks
    
'''

persist_directory = "db/chroma_db"

model = HuggingFaceEmbeddings(model="intfloat/e5-large-v2")

# Load the LLM model
hf_endpoint = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.1-8B-Instruct", # you have to make sure that this model has an InferenceProvider on the HuggingFace Website.
    task="text-generation",
    temperature=0,
    provider="auto"
)
llm = ChatHuggingFace(llm=hf_endpoint)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Pydantic model for structured output
# This is us basically saying: "Whatever the LLM outputs, I want it to look like a dictionary with a key called queries which holds a list of strings."
class QueryVariations(BaseModel):
    queries: List[str]

# Although it is mentioned in the documentation, the ChatHuggingFace does not currently support Pydantic models. So, we have to change the code a little.
# (I think ChatOpenAI does.)

# The JsonOutputParser is the first responder. It's job is to :
# 1. Generate instructions : It creates a string of text (for ex" Output must be a JSON with a queries key, etc.) to insert into the prompt.
# 2. When the LLM responses, it looks for a JSON block in the text and tries to turn it into a Python dictionary
# However, if the LLM adds something extra like : Sure, here is your JSON...., a standard parser crashes. 
base_parser = JsonOutputParser(pydantic_object=QueryVariations)

# This is the safety net. 
# If the JsonOutputParser fails, this parser doesn't just throw an error. 
# Instead, it takes the bad output, the error message, and the original prompt, and sends them back to the LLM saying: "You messed up the JSON. Here is the error. Please fix it."
# It effectively "self-heals" the data.
robust_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)


# input_variables: These are the variables you provide every time you run the code (like the og_query).
# partial_variables: These are "pre-filled" variables. Instead of manually typing "Please return JSON..." every time, you use get_format_instructions(). 
# This dynamically injects the exact technical requirements derived from your Pydantic model into the prompt.
prompt = PromptTemplate(
    template="Generate THREE different variations of this query. \n{format_instructions}\nOriginal: {og_query}. Return 3 alternative queries that rephrase or approach the same question from different angles",
    input_variables=["og_query"],
    partial_variables={"format_instructions": robust_parser.get_format_instructions()},
)


# LangChain uses LCEL (LangChain Expression Language). The pipe operator (|) works exactly like a literal assembly line:
    # Input: You pass {"og_query": "..."}.
    # prompt: Takes your query, merges it with the format instructions, and creates one big string.
    # llm: Receives that string, thinks, and spits out a raw text response (potentially messy).
    # robust_parser: Receives the raw text. It tries to parse it; if it's broken, it triggers the "Fixing" logic to ensure you get a clean Python object back.
chain = prompt | llm | robust_parser

response = chain.invoke({"og_query": "How does Tesla make money?" })
query_variations = response['queries']

retriever = db.as_retriever(search_kwargs={"k":5})

extracted_documents = []

for query in query_variations:
    relevant_docs = retriever.invoke(query)
    extracted_documents.append(relevant_docs)
    # print(f"Documents found for query : {query}")
    # for i, doc in enumerate(relevant_docs):
    #     print(f"        Document {i} - {doc.page_content[:300]}...")
    # print("-"*100)

rrf_outputs = reciprocal_rank_fusion(extracted_documents)