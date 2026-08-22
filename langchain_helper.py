from langchain_core.documents import Document

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate


def create_vectordb_from_transcript(transcript):

    document = Document(
        page_content=transcript
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    docs = text_splitter.split_documents(
        [document]
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        )
    )

    db = FAISS.from_documents(
        docs,
        embeddings
    )

    return db


def get_response_from_query(
    db,
    groq_api_key,
    query,
    k=4
):

    # Find the most relevant transcript chunks
    docs = db.similarity_search(
        query,
        k=k
    )

    docs_page_content = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2,
        api_key=groq_api_key
    )

    prompt = PromptTemplate(
        input_variables=[
            "question",
            "docs"
        ],
        template="""
You are a helpful assistant that answers
questions about a YouTube video based only
on the provided video transcript.

Question:
{question}

Video transcript:
{docs}

Instructions:
- Only use factual information from the transcript.
- If the transcript does not contain enough
  information, say "I don't know".
- Do not invent information.
- Give a clear and detailed answer.
"""
    )

    chain = prompt | llm

    response = chain.invoke({
        "question": query,
        "docs": docs_page_content
    })

    return response.content