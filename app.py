import streamlit as st
from scraper import scrape_website
from vector_store import build_vector_store, load_vector_store
from rag_chain import get_answer

st.set_page_config(page_title="WebMind AI", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }

    .main { background-color: #fafafa; }

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1e293b;
        text-align: center;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .status-card {
        background: #ecfdf5;
        border: 1px solid #6ee7b7;
        border-radius: 10px;
        padding: 10px 14px;
        margin-top: 10px;
        color: #047857;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .stChatMessage {
        border-radius: 14px !important;
        border: 1px solid #e2e8f0 !important;
        margin-bottom: 10px !important;
    }

    .stButton>button {
        background-color: #4f46e5 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }

    .stButton>button:hover {
        background-color: #4338ca !important;
        transform: translateY(-1px) !important;
    }

    .stTextInput>div>div>input {
        border-radius: 8px !important;
        border: 1px solid #cbd5e1 !important;
    }

    .stChatInput>div {
        border-radius: 14px !important;
        border: 1px solid #cbd5e1 !important;
    }

    .info-box {
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        color: #4338ca;
        margin: 1.5rem 0;
    }

    .feature-badge {
        display: inline-block;
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        color: #4338ca;
        margin: 4px;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding: 16px 0;'>
            <div style='font-size: 2.5rem;'>🧠</div>
            <div style='color: #1e293b; font-size: 1.3rem; font-weight: 700;'>WebMind AI</div>
            <div style='color: #94a3b8; font-size: 0.8rem;'>Powered by RAG + Groq</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='color:#64748b; font-size:0.85rem; font-weight:600;'>WEBSITE INDEXER</p>", unsafe_allow_html=True)

    url = st.text_input("", placeholder="https://example.com", label_visibility="collapsed")
    index_btn = st.button("Index Website")

    if index_btn and url:
        with st.spinner("Scraping and indexing..."):
            chunks = scrape_website(url)
            if chunks:
                build_vector_store(chunks)
                st.success(f"Indexed {len(chunks)} pages!")
                st.session_state.indexed_url = url
            else:
                st.error("Failed to scrape. Try another URL.")

    if "indexed_url" in st.session_state:
        st.markdown(f"""
            <div class='status-card'>
                ✓ Active: {st.session_state.indexed_url[:35]}...
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("LangChain • ChromaDB • Groq • Streamlit")

# Main area
st.markdown("<div class='hero-title'>WebMind AI</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Ask anything about any website — instantly</div>", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align:center; margin-bottom:1.5rem;'>
        <span class='feature-badge'>RAG Pipeline</span>
        <span class='feature-badge'>Web Scraping</span>
        <span class='feature-badge'>Vector Search</span>
        <span class='feature-badge'>LLM Answers</span>
    </div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("""
        <div class='info-box'>
            <div style='font-size:2rem;'>🌐</div>
            <div style='font-size:1.1rem; font-weight:600; margin:8px 0;'>No website indexed yet</div>
            <div style='font-size:0.9rem;'>Enter a URL in the sidebar and click Index Website to get started</div>
        </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Sources"):
                for s in set(msg["sources"]):
                    st.markdown(f"- [{s}]({s})")

if prompt := st.chat_input("Ask anything about the indexed website..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = get_answer(prompt)
        st.markdown(answer)
        if sources:
            with st.expander("View Sources"):
                for s in set(sources):
                    st.markdown(f"- [{s}]({s})")
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})