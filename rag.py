import os
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from groq import Groq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MAX_DIRECT_CONTEXT_CHARS = 20000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "web_assistant"
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
COMMENT_SECTION_MARKERS = [
    "Leave a Reply",
    "Leave a Comment",
    "Post Comment",
    "Your email address will not be published",
]
llm = None
vector_store = None
full_page_text = ""
full_page_sources = []

def get_groq_api_key():
    try:
        import _snowflake
        return _snowflake.get_generic_secret_string("groq_api_key")
    except ImportError:
        return os.getenv("GROQ_API_KEY")

def get_available_model(api_key):
    client = Groq(api_key=api_key)
    models = client.models.list()

    preferred_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b"
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
        "No compatible Groq text generation model is available for this API key."
    )

def strip_comment_section(text):
    lowered = text.lower()
    cut_index = len(text)

    for marker in COMMENT_SECTION_MARKERS:
        idx = lowered.find(marker.lower())
        if idx != -1:
            cut_index = min(cut_index, idx)

    return text[:cut_index].strip()

def fetch_url_as_document(url):
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to load {url}: server returned HTTP {response.status_code}. "
            "The site is likely blocking automated requests from this server."
        )

    soup = BeautifulSoup(response.text, "lxml")

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    raw_text = soup.get_text(separator="\n")
    lines = [line.strip() for line in raw_text.splitlines()]
    text = "\n".join(line for line in lines if line)

    return Document(page_content=text, metadata={"source": url})

def initialize_components():
    global llm, vector_store

    api_key = get_groq_api_key()

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Set it in .env locally, or as a "
            "secret named GROQ_API_KEY in your deployment platform."
        )

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

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR)
        )

def process_urls(urls):
    global full_page_text, full_page_sources

    yield "initializing components..."
    initialize_components()

    yield "Resetting vector store"
    vector_store.reset_collection()

    yield "loading the data from URLs"
    documents = []
    for url in urls:
        documents.append(fetch_url_as_document(url))

    if not documents or all(len(doc.page_content.strip()) < 200 for doc in documents):
        raise RuntimeError(
            "Could not extract usable content from the provided URL(s). "
            "The page loaded but returned little or no readable text."
        )

    for doc in documents:
        doc.page_content = strip_comment_section(doc.page_content)

    full_page_text = "\n\n".join(
        f"[Source: {doc.metadata.get('source', urls[0])}]\n{doc.page_content}"
        for doc in documents
    )
    full_page_sources = list(urls)

    yield "Splitting data into small chunks..."
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    docs = splitter.split_documents(documents)

    for i, doc in enumerate(docs):
        doc.metadata["source"] = urls[i % len(urls)]

    yield "Adding doc chunks into chromaDB..."

    ids = [str(uuid4()) for _ in range(len(docs))]

    vector_store.add_documents(
        docs,
        ids=ids
    )

    yield "Vector store successfully updated"

def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database is empty")

    if full_page_text and len(full_page_text) <= MAX_DIRECT_CONTEXT_CHARS:
        content = full_page_text
        sources = "\n".join(full_page_sources)
    else:
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        docs = retriever.invoke(query)
        content = "\n\n".join([doc.page_content for doc in docs])
        sources = "\n".join(
            list(
                set(
                    [
                        doc.metadata.get("source", "")
                        for doc in docs
                    ]
                )
            )
        )

    prompt = f"""
    You are a helpful assistant answering questions strictly from the content below,
    which was scraped from one or more web pages. The content may contain a numbered list.
    If the question refers to a position in a list (first, second, last, number N) or asks
    for a count, use the numbering exactly as it appears in the content to answer precisely.
    If the answer is not present in the content, say "I don't know". Don't hallucinate.

    content:
    {content}

    question:
    {query}
    """

    response = llm.invoke(prompt)

    return response.content.strip(), sources
