from pathlib import Path

from langchain_chroma import Chroma

from backend.rag.embeddings import get_embeddings


PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "db"

VECTOR_STORE = None


def load_vector_store():
    """
    Load the Chroma vector database once and reuse it.
    """

    global VECTOR_STORE

    if VECTOR_STORE is None:

        VECTOR_STORE = Chroma(
            persist_directory=str(DB_PATH),
            embedding_function=get_embeddings()
        )

    return VECTOR_STORE



