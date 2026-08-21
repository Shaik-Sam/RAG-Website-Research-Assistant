import os
import re
from uuid import uuid4
from pathlib import Path

import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources" / "vectorstore"
COLLECTION_NAME = "web_assistant"

llm = None
vector_store = None
full_page_text = ""
full_page_sources = []


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

    raise RuntimeError(
        "GROQ_API_KEY is not configured. Add it to Streamlit Cloud Secrets."
    )


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
        "distil-whisper"
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
            temperature=0.2,
            max_tokens=2000,
            api_key=api_key
        )

    if vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                "trust_remote_code": True
            }
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
        ),
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30,
        allow_redirects=True
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
            "svg",
            "iframe",
            "nav",
            "footer",
            "header"
        ]
    ):
        element.decompose()

    main = soup.find("article")

    if main is None:
        main = soup.find("main")

    if main is None:
        main = soup.body

    if main is None:
        raise RuntimeError(
            "No readable content was found on the webpage."
        )

    text = main.get_text(
        separator="\n",
        strip=True
    )

    lines = []

    for line in text.splitlines():
        line = re.sub(
            r"\s+",
            " ",
            line
        ).strip()

        if line:
            lines.append(line)

    text = "\n".join(lines)

    if len(text) < 100:
        raise RuntimeError(
            "The webpage returned insufficient readable content."
        )

    return text


def process_urls(urls):
    global full_page_text
    global full_page_sources

    yield "Initializing components..."

    initialize_components()

    yield "Resetting vector store..."

    vector_store.reset_collection()

    yield "Loading data from URLs..."

    loaded_pages = []

    for url in urls:
        try:
            text = load_webpage(url)

            loaded_pages.append(
                {
                    "url": url,
                    "text": text
                }
            )

            yield f"Successfully loaded: {url}"

        except requests.exceptions.RequestException as error:
            raise RuntimeError(
                f"Could not access {url}: {error}"
            )

        except Exception as error:
            raise RuntimeError(
                f"Could not extract content from {url}: {error}"
            )

    if not loaded_pages:
        raise RuntimeError(
            "No webpage content could be loaded."
        )

    full_page_sources = [
        page["url"]
        for page in loaded_pages
    ]

    full_page_text = "\n\n".join(
        f"[Source: {page['url']}]\n{page['text']}"
        for page in loaded_pages
    )

    yield "Splitting data into small chunks..."

    splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            ". ",
            ".",
            " ",
            ""
        ],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    docs = []

    for page in loaded_pages:
        page_docs = splitter.create_documents(
            [page["text"]]
        )

        for index, doc in enumerate(page_docs):
            doc.metadata["source"] = page["url"]
            doc.metadata["chunk_index"] = index
            docs.append(doc)

    if not docs:
        raise RuntimeError(
            "No document chunks were created."
        )

    yield f"Created {len(docs)} document chunks..."

    ids = [
        str(uuid4())
        for _ in docs
    ]

    yield "Adding document chunks into ChromaDB..."

    vector_store.add_documents(
        documents=docs,
        ids=ids
    )

    yield "Vector store successfully updated."


def get_all_stored_documents():
    data = vector_store.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    documents = data.get(
        "documents",
        []
    )

    metadatas = data.get(
        "metadatas",
        []
    )

    result = []

    for document, metadata in zip(
        documents,
        metadatas
    ):
        result.append(
            {
                "text": document,
                "metadata": metadata or {}
            }
        )

    return result


def get_requested_number(query):
    query_lower = query.lower()

    numeric_match = re.search(
        r"\b(?:quote|item|number)\s*#?\s*(\d+)\b",
        query_lower
    )

    if numeric_match:
        return int(
            numeric_match.group(1)
        )

    ordinal_numbers = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10,
        "eleventh": 11,
        "twelfth": 12,
        "thirteenth": 13,
        "fourteenth": 14,
        "fifteenth": 15,
        "sixteenth": 16,
        "seventeenth": 17,
        "eighteenth": 18,
        "nineteenth": 19,
        "twentieth": 20
    }

    for word, number in ordinal_numbers.items():
        if re.search(
            rf"\b{word}\b",
            query_lower
        ):
            return number

    return None


def extract_numbered_quote(query):
    if (
        "quote" not in query.lower()
        and "quotes" not in query.lower()
    ):
        return None

    number = get_requested_number(query)

    if number is None:
        return None

    pattern = re.compile(
        rf"(?m)^\s*{number}\s*[\.\):-]\s*(.+?)(?=\n\s*\d+\s*[\.\):-]|\Z)",
        re.DOTALL
    )

    match = pattern.search(
        full_page_text
    )

    if match:
        quote = match.group(1).strip()

        if quote:
            return f"{number}. {quote}"

    return None


def extract_first_numbered_item():
    pattern = re.compile(
        r"(?m)^\s*1\s*[\.\):-]\s*(.+?)(?=\n\s*2\s*[\.\):-]|\Z)",
        re.DOTALL
    )

    match = pattern.search(
        full_page_text
    )

    if match:
        quote = match.group(1).strip()

        if quote:
            return f"1. {quote}"

    return None


def retrieve_documents(query):
    stored_documents = get_all_stored_documents()

    if not stored_documents:
        return []

    direct_quote = extract_numbered_quote(
        query
    )

    if direct_quote:
        return [
            {
                "text": direct_quote,
                "metadata": {
                    "source": full_page_sources[0]
                    if full_page_sources
                    else ""
                }
            }
        ]

    if (
        "first quote" in query.lower()
        or "first quotation" in query.lower()
        or "quote number 1" in query.lower()
        or "quote #1" in query.lower()
    ):
        first_quote = extract_first_numbered_item()

        if first_quote:
            return [
                {
                    "text": first_quote,
                    "metadata": {
                        "source": full_page_sources[0]
                        if full_page_sources
                        else ""
                    }
                }
            ]

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

    docs = retriever.invoke(
        query
    )

    results = []

    for doc in docs:
        results.append(
            {
                "text": doc.page_content,
                "metadata": doc.metadata
            }
        )

    return results


def build_context(documents):
    context_parts = []

    for index, document in enumerate(
        documents,
        start=1
    ):
        context_parts.append(
            f"""
SOURCE {index}
URL: {document["metadata"].get("source", "")}

{document["text"]}
"""
        )

    return "\n".join(
        context_parts
    )


def generate_answer(query):
    initialize_components()

    documents = retrieve_documents(
        query
    )

    if not documents:
        return (
            "I could not find relevant information in the processed webpage.",
            "\n".join(full_page_sources)
        )

    context = build_context(
        documents
    )

    prompt = f"""
You are a website research assistant.

Answer the user's question using only the website content provided below.

Rules:
1. Answer the question directly.
2. Do not invent information.
3. If the answer exists in the content, provide it.
4. If the user asks for a numbered quote, return the requested quote exactly as it appears in the content.
5. If the user asks for the first quote, return quote number 1.
6. If the user asks for a specific quote number, return that numbered quote.
7. If the answer genuinely cannot be found, say that it was not found in the provided webpage.
8. Never answer with "0" unless the webpage itself explicitly contains 0 as the answer.
9. Preserve important wording from quotes.

WEBSITE CONTENT:

{context}

USER QUESTION:

{query}

ANSWER:
"""

    response = llm.invoke(
        prompt
    )

    answer = response.content.strip()

    if not answer:
        answer = (
            "I could not generate an answer from the provided webpage."
        )

    sources = []

    for document in documents:
        source = document["metadata"].get(
            "source",
            ""
        )

        if source and source not in sources:
            sources.append(
                source
            )

    if not sources:
        sources = full_page_sources

    return (
        answer,
        "\n".join(sources)
    )
