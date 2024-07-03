from langchain_postgres import PGVector
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings


connection = "postgresql+psycopg://langchain:langchain@13.246.58.40:6024/langchain"
collection_name = "arxiv_docs"
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = PGVector(
    embeddings = embeddings,
    collection_name = collection_name, 
    connection = connection,
    use_jsonb = True
)

print("Performing similarity search...")
print(vectorstore.similarity_search("C#", k=2))