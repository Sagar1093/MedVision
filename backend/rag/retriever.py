from pathlib import Path
from langchain_chroma import Chroma
from backend.rag.embeddings import get_embeddings

PROJECT_ROOT = Path(__file__).resolve().parent

DB_PATH = PROJECT_ROOT/"db"
VECTOR_STORE = None
def load_vector_store():
    global VECTOR_STORE

    if VECTOR_STORE is None:
        print("Loading Chroma DB")

        VECTOR_STORE = Chroma(
            persist_directory=str(DB_PATH),
            embedding_function=get_embeddings()
        )
        print("Chroma DB loaded")

    return VECTOR_STORE

def get_retriever():
    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_kwargs={"k":4}
    )
    return retriever


