from langchain_core.prompts import ChatPromptTemplate
from backend.rag.retriever import load_vector_store
from backend.rag.llm import get_llm

PROMPT = """
You are MedVision AI, an intelligent medical assistant.

Answer ONLY using the retrieved medical context.

Rules:
- Use only the retrieved context.
- Never make up medical facts.
- If the answer is not present in the context, say you don't know.
- If the user refers to "this disease", "this image", or "the predicted disease",
  interpret it using the CNN prediction.
- Do not provide a definitive medical diagnosis.
- Cite important statements using [Source X].
- Finish with an educational disclaimer.

CNN Prediction:
{prediction}

Retrieved Context:
{context}

User Question:
{question}
"""


def retrieve_context(question, disease=None, k=5):
    """
    Retrieve the most relevant chunks from Chroma.
    Optionally filter by disease.
    """

    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
            search_type = "mmr",
            search_kwargs = {
                            "k":5,
                            "fetch_k":15,
                            "lambda_mult":0.7
                            
                        }
           
        )

    docs = retriever.invoke(question)


    return docs




def ask_question(question,prediction=None):
    """Full RAG Pipeline"""

    docs = retrieve_context(question)
    
    prompt = ChatPromptTemplate.from_template(PROMPT)

    context_parts = []

    sources = []
    seen = set()

    for i, doc in enumerate(docs, start=1):

        source = doc.metadata["source"]
        page = doc.metadata["page"] + 1

        context_parts.append(
            f"""
            Source {i}
            File: {source}
            Page: {page}

            {doc.page_content}
            """
        )

        if (source, page) not in seen:
            seen.add((source, page))

            sources.append({
                "id": f"Source {i}",
                "source": source,
                "page": page
            })

    context = "\n\n----------\n\n".join(context_parts)

    llm = get_llm()

    response = llm.invoke(
            prompt.format(
                prediction=prediction,
                context=context,
                question=question
            )
        )

    content = response.content

    if isinstance(content,list):
        answer = ""

        for block in content:
            if isinstance(block,dict) and block.get("type") == "text":
                answer += block["text"]
    else:
        answer = content

    return {
        "answer":answer,
        "sources":sources
    }


def stream_question(question, prediction=None):
    """Stream the Gemini response while using the same RAG pipeline."""

    docs = retrieve_context(
        question=question
    )

    context_parts = []

    for i, doc in enumerate(docs, start=1):

        source = doc.metadata["source"]
        page = doc.metadata["page"] + 1

        context_parts.append(
            f"""
            Source {i}
            File: {source}
            Page: {page}

            {doc.page_content}
            """
        )

    context = "\n\n----------\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_template(PROMPT)

    llm = get_llm()

    for chunk in llm.stream(
        prompt.format(
            prediction=prediction,
            context=context,
            question=question
        )
    ):
        if chunk.content:

            if isinstance(chunk.content, list):

                for block in chunk.content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                    ):
                        yield block["text"]

            else:
                yield chunk.content

if __name__ == "__main__":
    result = ask_question(question="how did cnn predict the disease",
                          prediction="tuberculosis")

    print("="*100)
    print(result["answer"])

    print()

    for source in result["sources"]:
        print(f"[{source['id']}]"
              f"{source['source']}"
              f"(Page {source['page']})")
