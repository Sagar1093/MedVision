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


def retrieve_context(question, k=4):
    """
    Retrieve the most relevant and diverse chunks from Chroma using MMR.
    """

    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 15,
            "lambda_mult": 0.7
        }
    )

    return retriever.invoke(question)


def prepare_context(docs):
    """
    Prepare retrieved documents and source information
    for the LLM.
    """

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

            sources.append(
                {
                    "id": f"Source {i}",
                    "source": source,
                    "page": page
                }
            )

    context = "\n\n----------\n\n".join(
        context_parts
    )

    return context, sources


def extract_content(content):
    """
    Extract text from Gemini's response content.
    """

    if isinstance(content, list):

        answer = ""

        for block in content:

            if (
                isinstance(block, dict)
                and block.get("type") == "text"
            ):
                answer += block["text"]

        return answer

    return content


def ask_question(question, prediction=None):
    """
    Run the complete RAG pipeline and return
    the final answer with sources.
    """

    docs = retrieve_context(question)

    context, sources = prepare_context(docs)

    prompt = ChatPromptTemplate.from_template(
        PROMPT
    )

    llm = get_llm()

    response = llm.invoke(
        prompt.format(
            prediction=prediction,
            context=context,
            question=question
        )
    )

    answer = extract_content(
        response.content
    )

    return {
        "answer": answer,
        "sources": sources
    }


def stream_question(question, prediction=None):
    """
    Stream the Gemini response while returning
    retrieved sources.
    """

    docs = retrieve_context(question)

    context, sources = prepare_context(docs)

    prompt = ChatPromptTemplate.from_template(
        PROMPT
    )

    llm = get_llm()


    yield {
        "type": "sources",
        "sources": sources
    }

    # Stream Gemini response.
    for chunk in llm.stream(
        prompt.format(
            prediction=prediction,
            context=context,
            question=question
        )
    ):

        if not chunk.content:
            continue

        if isinstance(chunk.content, list):

            for block in chunk.content:

                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                ):
                    yield {
                        "type": "token",
                        "content": block["text"]
                    }

        else:

            yield {
                "type": "token",
                "content": chunk.content
            }


    yield {
        "type": "done"
    }