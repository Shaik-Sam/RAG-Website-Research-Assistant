# RAG-Website-Research-Assistant
Retrieval-Augmented Generation (RAG) application designed to act as a "Website Research Assistant." It allows users to input multiple URLs, scrape their content, and interact with that data through a conversational interface powered by a Large Language Model (LLM).

# RAG-Based Website Research Assistant 

An intelligent Retrieval-Augmented Generation (RAG) system that allows you to "chat" with any website. Built with **LangChain**, **Groq (Llama 3.3)**, and **ChromaDB**.

##  Overview
This project enables users to extract and query information from multiple URLs simultaneously. By combining high-speed LLM inference with a local vector database, it provides a fast and accurate way to research online documentation, articles, or blogs.

##  Tech Stack
- **Frontend:** [Streamlit](https://streamlit.io/)
- **Orchestration:** [LangChain](https://www.langchain.com/)
- **LLM:** Meta Llama 3.3-70b via [Groq Cloud](https://groq.com/)
- **Vector Database:** [ChromaDB](https://www.trychroma.com/)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
- **Data Loading:** Unstructured URL Loader

##  Features
- **Multi-URL Support:** Process up to 3 URLs at once.
- **Efficient Chunking:** Smart text splitting for better context retention.
- **Source Attribution:** The assistant cites exactly which URL it used to generate an answer.
- **High Speed:** Powered by Groq for near-instant responses.

##  Installation & Setup
1. **Clone the repo:**
   ```bash
   git clone (https://github.com/Shaik-Sam/RAG-Website-Research-Assistant.git)

2. **Install dependencies:** pip install -r requirements.txt
3. **Environment Variables:**
Create a .env file in the root directory and add your API key: GROQ_API_KEY=your_groq_api_key_here
4. **Run the App:** streamlit run main.py
