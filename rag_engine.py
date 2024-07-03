from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_postgres import PGVector

connection = "postgresql+psycopg://langchain:langchain@13.246.58.40:6024/langchain"
collection_name = "arxiv_docs"
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


vectorstore = PGVector(
    embeddings = embeddings,
    collection_name = collection_name, 
    connection = connection,
    use_jsonb = True
)

#Initialize LLM.
llm = HuggingFacePipeline.from_model_id(model_id="", task="text2text-generation", model_kwargs={"temperature": 0, "max_length": 200,  "max_new_tokens":512,
    "top_k":10, "top_p":0.95, "typical_p":0.95,
    "temperature":0.01, "repetition_penalty":1.03,}, device=0)

def similarity_search(text, num_docs):
    '''Perform similarity search from query and return relevant documents.'''
    documents = vectorstore.similarity_search(text, k = num_docs)
    return documents


def rag_function(query, num_docs=5):
    # Perform similarity search
    relevant_docs = similarity_search(query, num_docs)
    
    # Concatenate the content of the relevant documents
    context = " ".join([doc.page_content for doc in relevant_docs])

    # Create a prompt for the LLM
    messages = [
        {"role": "system", "content": "Summarize the text below for a second-grade student."},
        {"role": "user", "content": context}
    ]
    
    # Use the LLM to generate a response
    response = llm.generate(messages)
    
    return response

# Example usage
query = "Explain the theory of relativity."
output = rag_function(query)
print(output)

prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
2. If you find the answer, write the answer in a concise way with five sentences maximum.

{context}

Question: {question}

Helpful Answer:
"""

PROMPT = PromptTemplate(
 template=prompt_template, input_variables=["context", "question"]
)

retrievalQA = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# Call the QA chain with our query.
result = retrievalQA.invoke({"query": query})
print(result['result'])