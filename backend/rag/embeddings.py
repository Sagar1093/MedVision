from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "BAAI/bge-small-en-v1.5"

def get_embeddings():
    """Returns the embedding model used throughout the RAG pipeline"""

    embeddings = HuggingFaceEmbeddings(
        model_name = MODEL_NAME,
        model_kwargs = {"device":"cpu"},
        encode_kwargs = {"normalize_embeddings":True}
    )

    return embeddings

