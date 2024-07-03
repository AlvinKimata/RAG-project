from warnings import filterwarnings
filterwarnings('ignore')

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_postgres import PGVector


connection = "postgresql+psycopg://langchain:langchain@13.246.58.40:6024/langchain"
collection_name = "arxiv_docs"

print("Loading Sentence transformer and vectorstore...")

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=collection_name, 
    connection=connection,
    use_jsonb=True
)

print("Loaded Sentence transformer and vectorstore.")


def rag_function(query, num_docs=3):
    # Perform similarity search
    relevant_docs = similarity_search(query, num_docs)
    
    # Concatenate the content of the relevant documents
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    return context

def similarity_search(text, num_docs):
    '''Perform similarity search from query and return relevant documents.'''
    documents = vectorstore.similarity_search(text, k=num_docs)
    return documents


# Example usage
question = "Object oriented principles in C#"
context = rag_function(question)

prompt_template = f"""
You are a helpful assistant. Use the information provided to answer the question below. Follow these rules:
1. Base your answer on the facts in the provided information.
2. Keep your answer concise, up to five sentences.
3. If the information doesn't contain the answer, inform the user and recommend relevant articles.

Information:\n {context}

Question: {question}

Helpful Answer:
"""


print(f"template: {prompt_template}")
