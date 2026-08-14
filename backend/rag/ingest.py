from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from backend.rag.embeddings import get_embeddings
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent

DOCUMENTS_PATH = PROJECT_ROOT/"documents"

DB_PATH = PROJECT_ROOT/"db"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)



def load_documents():
    pdf_files = list(DOCUMENTS_PATH.rglob("*.pdf"))
    documents = []
    
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        disease = pdf_path.parent.name
        source = pdf_path.name
        

    

        

        for page in pages:
            page.metadata["disease"] = disease
            page.metadata["source"] = source
    

        documents.extend(pages)

    return documents

def split_documents(documents):
    chunks = text_splitter.split_documents(documents)

    print(f"created{len(chunks)} chunks")

    return chunks

def build_vector_store(chunks):
    if DB_PATH.exists():
        shutil.rmtree(DB_PATH)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(DB_PATH)
    )

    print("Vector Database created successfully")
    return vector_store

def main():
    documents = load_documents()

    chunks = split_documents(documents)

    build_vector_store(chunks)

    print("RAG Pipeline completed")
if __name__=="__main__":
    main()