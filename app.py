import streamlit as st
from scraper import scrape_website
from vector_store import build_vector_store, load_vector_store
from rag_chain import get_answer

st.set_page_config(page_title="RAG Website Chatbot", layout="centered")
st.title("RAG-Powered Website Chatbot")
st.caption("Ask anything about a website - powered by LLM + RAG")

with st.sidebar:
    st.header("Setup")
    url = st.text_input("Enter website URL to index:", placeholder="https://example.com")
    index_btn = st.button("Index Website")
    if index_btn and url:
        with st.spinner("Scraping and indexing..."):
            chunks = scrape_website(url)
            if chunks:
                build_vector_store(chunks)
                st.success(f"Indexed {len(chunks)} chunks!")
            else:
                st.error("Failed to scrape the URL. Try another.")

if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
if prompt := st.chat_input("Ask a question about the website..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = get_answer(prompt)
        st.markdown(answer)
        if sources:
            st.caption(f"Sources: {', '.join(set(sources))}")
    st.session_state.messages.append({"role": "assistant", "content": answer})