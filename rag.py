import os
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
import requests
import streamlit as st
from bs4 import BeautifulSoup
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()

CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "web_assistant"

llm = None
vector_store = None


def get_groq_api_key():
    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        return api_key.strip()

    try:
        api_key = st.secrets["GROQ_API_KEY"]

        if api_key:
            return str(api_key).strip()
    except Exception:
        pass

    raise RuntimeError("GROQ_API_KEY is not configured.")


def get_available_model(api_key):
    client = Groq(api_key=api_key)
    models = client.models.list()

    preferred_models = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant"
    ]

    available_models = [
        model.id
        for model in models.data
        if getattr(model, "active", True)
    ]

    for model in preferred_models:
        if model in available_models:
            return model

    excluded = [
        "whisper",
        "guard",
        "tts",
        "speech",
        "audio"
    ]

    for model in available_models:
        if not any(
            word in model.lower()
            for word in excluded
        ):
            return model

    raise RuntimeError(
        "No compatible Groq text generation model is available."
    )


def initialize_components():
    global llm
    global vector_store

    api_key = get_groq_api_key()

    if llm is None:
        model_name = get_available_model(api_key)

        llm = ChatGroq(
            model=model_name,
            temperature=0.9,
            max_tokens=1000,
            api_key=api_key
        )

    if vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        VECTORSTORE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR)
        )


def load_webpage(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    for element in soup.find_all(
        [
            "script",
            "style",
            "noscript",
            "svg"
        ]
    ):
        element.decompose()

    content = soup.find("article")

    if content is None:
        content = soup.find("main")

    if content is None:
        content = soup.body

    if content is None:
        raise RuntimeError(
            "No readable content was found on the webpage."
        )

    text = content.get_text(
        separator="\n",
        strip=True
    )

    lines = []

    for line in text.splitlines():
        line = " ".join(
            line.split()
        )

        if line:
            lines.append(line)

    text = "\n".join(lines)

    if len(text.strip()) < 100:
        raise RuntimeError(
            "The webpage did not return enough readable content."
        )

    return text


def process_urls(urls):
    yield "initializing components..."

    initialize_components()

    yield "Resetting vector store"

    vector_store.reset_collection()

    yield "loading the data from URLs"

    documents = []

    for url in urls:
        try:
            text = load_webpage(url)

            from langchain_core.documents import Document

            document = Document(
                page_content=text,
                metadata={
                    "source": url
                }
            )

            documents.append(document)

        except Exception as error:
            raise RuntimeError(
                f"Could not load content from {url}: {error}"
            )

    if not documents:
        raise RuntimeError(
            "Could not extract usable content from the provided URL(s)."
        )

    yield "Splitting data into small chunks..."

    splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            ".",
            " "
        ],
        chunk_size=CHUNK_SIZE
    )

    docs = splitter.split_documents(
        documents
    )

    for i, doc in enumerate(docs):
        doc.metadata["source"] = urls[
            i % len(urls)
        ]

    yield "Adding doc chunks into chromaDB..."

    ids = [
        str(uuid4())
        for _ in range(len(docs))
    ]

    vector_store.add_documents(
        docs,
        ids=ids
    )

    yield "Vector store successfully updated"


def generate_answer(query):
    if vector_store is None:
        raise RuntimeError(
            "Vector database is empty"
        )

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 6
        }
    )

    docs = retriever.invoke(
        query
    )

    if not docs:
        return (
            "I don't know.",
            ""
        )

    content = "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )

    prompt = f"""
You are a helpful assistant.

Answer the question using the content given below.

If the answer is present in the content, provide the answer clearly.

If the user asks for a specific item, quote, number, person, fact, or detail, find it from the provided content.

If the answer is not present, say "I don't know".

Do not hallucinate or invent information.

content:
{content}

question:
{query}
"""

    response = llm.invoke(
        prompt
    )

    sources = "\n".join(
        list(
            set(
                [
                    doc.metadata.get(
                        "source",
                        ""
                    )
                    for doc in docs
                ]
            )
        )
    )

    return (
        response.content.strip(),
        sources
    )
