import os
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq  # Use Groq, not Google
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Variables
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "web_assistant"
llm = None
vector_store = None

def initialize_components():
    global llm, vector_store

    if llm is None:
        # We explicitly use ChatGroq and fetch the key from your .env
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.9,
            max_tokens=1000,
            api_key=os.getenv("GROQ_API_KEY") 
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
    yield "initializing components..."
    initialize_components()

    yield "Resetting vector store"
    vector_store.reset_collection()

    yield "loading the data from URLs"
    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()

    yield "Splitting data into small chunks..."
    splitter=RecursiveCharacterTextSplitter(
        separators=["\n\n","\n","."," "],
        chunk_size=CHUNK_SIZE
    )

    docs=splitter.split_documents(documents)

    # attaching urls(sources) to each document
    for i,doc in enumerate(docs):
        doc.metadata["source"]=urls[i%len(urls)]
    yield "Adding doc chunks into chromaDB..."

    ids=[str(uuid4())for _ in range(len(docs))]

    vector_store.add_documents(docs,ids=ids)

    yield "Vector store successfully updated"



def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database is empty")
    retriever= vector_store.as_retriever()

    docs=retriever.invoke(query)

    content="\n\n".join([doc.page_content for doc in docs])

    prompt=f"""
    You are a helpful  assistant .
    Answer the question using the content given below.
    if the answer is not present,say "I don't know".Don't hallucinate.

    content:
    {content}

    question:
    {query}
    """

    response=llm.invoke(prompt)

    sources="\n".join(
        list(set([doc.metadata.get("source","") for doc in docs]))
    )

    return response.content.strip(),sources


