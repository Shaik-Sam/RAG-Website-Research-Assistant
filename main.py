import streamlit as st
from rag import process_urls, generate_answer

st.title("RAG Website Research Assistant")

st.write(
    "Enter one or more website URLs, process their content, "
    "and ask questions about the information using RAG."
)

url1 = st.sidebar.text_input("URL-1")
url2 = st.sidebar.text_input("URL-2")
url3 = st.sidebar.text_input("URL-3")

placeholder = st.empty()

process_url_button = st.sidebar.button("Process URLs")

if process_url_button:
    urls = [url for url in (url1, url2, url3) if url != ""]

    if len(urls) == 0:
        placeholder.warning("Please enter at least one URL.")
    else:
        for status in process_urls(urls):
            placeholder.info(status)

query = placeholder.text_input("Question")

if query:
    try:
        answer, sources = generate_answer(query)

        st.header("Answer")
        st.write(answer)

        if sources:
            st.subheader("Sources")

            for source in sources.split("\n"):
                st.write(source)

    except RuntimeError:
        st.warning("Please click 'Process URLs' before asking a question.")
