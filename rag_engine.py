from warnings import filterwarnings
filterwarnings('ignore')

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_postgres import PGVector

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
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

def similarity_search(text, num_docs):
    '''Perform similarity search from query and return relevant documents.'''
    documents = vectorstore.similarity_search(text, k=num_docs)
    return documents


def document_template(document):
    '''Returns a template with article title, id and abstract.'''
    abstract = document.page_content
    id = document.metadata['arXivId']
    title = document.metadata['title']
    template = f"Article title: {title}\narXivId: {id}\nAbstract:{abstract}"
    return template

def rag_function(query, num_docs=3):
    '''Perform similarity search and return the context.'''
    relevant_docs = similarity_search(query, num_docs)
    context = "\n\n".join([document_template(doc) for doc in relevant_docs])
    return context

def generate_prompt(query, num_docs=3):
    '''Generate the prompt using the given query.'''
    context = rag_function(query, num_docs)
    prompt_template = f"""
    You are a helpful assistant. Use the information provided to answer the question below. Follow these rules:
    1. Base your answer on the facts in the provided information.
    2. Keep your answer concise.
    3. Recommend an article relevant to the question based on title and arXiv ID from the context.
    4. If the information doesn't contain the answer, inform the user and recommend an article relevant to the question based on title and arXiv ID from the context.

    Question: 
    {query}

    Context:
    {context}

    Helpful Answer:
    """

    return prompt_template.strip()

# Example usage
question = "Provide examples of compiler optimization techniques."
prompt = generate_prompt(question)

print(f"template: {prompt}")
