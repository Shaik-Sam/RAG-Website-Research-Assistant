import streamlit as st
from  rag import process_urls,generate_answer 

st.title("RAG based Website Assistant")

url1 = st.sidebar.text_input("URL-1")
url2 = st.sidebar.text_input("URL-2")
url3 = st.sidebar.text_input("URL-3")

placeholder = st.empty()

process_url_button = st.sidebar.button("Process URLs")

if process_url_button:
    urls = [url for url in (url1,url2,url3) if url!='']

    if len(urls)==0:
        placeholder.text('Please Enter atleast one URL')
    else:
        for status in process_urls(urls): #function can store the data in chromaDB in small chunks in vector format
            placeholder.text(status)

query=placeholder.text_input("Question")

if query:
    try:
        answer,sources = generate_answer(query) #Gives answer from the chromaDB
        st.header("Answer")
        st.write(answer)
        if sources:
            st.subheader("Sources")
            for source in sources.split("\n"):
                st.write(source)
    except RuntimeError as e:
        placeholder.text("Please Click on Process URL button first")

