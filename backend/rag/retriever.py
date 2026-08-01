from pathlib import Path
from langchain_chroma import Chroma
from backend.rag.embeddings import get_embeddings

PROJECT_ROOT = Path(__file__).resolve().parent

DB_PATH = PROJECT_ROOT/"db"

def load_vector_store():
    vector_store = Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=get_embeddings()
    )

    return vector_store

def get_retriever():
    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_kwargs={"k":4}
    )
    return retriever

if __name__=="__main__":
    retriever = get_retriever()

    docs = retriever.invoke(
        "What are the symptoms of pneumonia"
    )
    for i,doc in enumerate(docs):
        print("="*60)

        print(doc.metadata)
        print()
        print(doc.page_content[:300])
