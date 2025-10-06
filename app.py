import streamlit as st
from datetime import datetime
from src.rag_system import PDFNotesRAG

st.set_page_config(page_title="Study Notes Assistant", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    /* General chat bubble styles */
    .chat-message {
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 10px;
        line-height: 1.6;
        font-size: 16px;
        word-wrap: break-word;
    }

    /* User message bubble */
    .user {
        background-color: #d9fdd3; /* WhatsApp-like green */
        color: #000000;
        text-align: right;
        border-top-right-radius: 0px;
    }

    /* Bot message bubble (adaptive) */
    .bot {
        background-color: #2b2b2b; /* dark gray for dark mode */
        color: #f5f5f5;
        text-align: left;
        border: 1px solid #444;
        border-top-left-radius: 0px;
    }

    /* When Streamlit is in light mode, adjust automatically */
    @media (prefers-color-scheme: light) {
        .bot {
            background-color: #ffffff;
            color: #222222;
            border: 1px solid #E4E4E4;
        }
    }

    .source-box {
        background-color: #f8f9fa;
        border-left: 4px solid #10A37F;
        padding: 8px 14px;
        border-radius: 8px;
        margin: 5px 0 10px 0;
        font-size: 14px;
        color: #333;
    }

    .main-title {
        text-align: center;
        font-size: 30px;
        font-weight: 700;
        color: #10A37F;
        margin-bottom: 4px;
    }

    .sub-title {
        text-align: center;
        color: #666;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


if "rag" not in st.session_state:
    st.session_state.rag = None
if "chat" not in st.session_state:
    st.session_state.chat = []
if "setup_done" not in st.session_state:
    st.session_state.setup_done = False

st.markdown('<div class="main-title"> Study Notes Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ask questions about your TSA study notes</div>', unsafe_allow_html=True)

def setup_rag():
    try:
        with st.spinner(" Setting up your AI study assistant..."):
            rag = PDFNotesRAG("./data")
            rag.load_pdfs()
            rag.chunk_documents()
            rag.setup_vector_store()
            rag.setup_gemini_llm("AIzaSyBpCOIHt6VO-OVj9pN8_PZC6oKtvlE14FI")
        st.session_state.rag = rag
        st.session_state.setup_done = True
        st.success(" System ready! Start chatting below.")
    except Exception as e:
        st.error(f" Setup failed: {e}")

if not st.session_state.setup_done:
    setup_rag()

chat_container = st.container()
with chat_container:
    for entry in st.session_state.chat:
        if entry["role"] == "user":
            st.markdown(f'<div class="chat-message user"> <b>You:</b><br>{entry["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot"> <b>AI:</b><br>{entry["content"]}</div>', unsafe_allow_html=True)
            if entry.get("sources"):
                sources_html = "".join(
                    [f"<div> Source {i+1}: Unit {src.metadata.get('unit', 'Unknown')} - Page {src.metadata.get('page', 0) + 1}</div>"
                     for i, src in enumerate(entry["sources"])]
                )
                st.markdown(f'<div class="source-box">{sources_html}</div>', unsafe_allow_html=True)

st.markdown("---")
col1, col2 = st.columns([4, 1])
with col1:
    prompt = st.text_input(" Type your question:", key="user_input", placeholder="e.g. Explain types of machine learning...")
with col2:
    send = st.button(" Send", use_container_width=True)

clear = st.button(" Clear Chat")

if clear:
    st.session_state.chat = []
    st.experimental_rerun()

if send and prompt.strip():
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.spinner(" Thinking..."):
        try:
            result = st.session_state.rag.ask_question(prompt)
            answer = result.get("answer", "No answer found.")
            sources = result.get("sources", [])
            st.session_state.chat.append({"role": "bot", "content": answer, "sources": sources})
        except Exception as e:
            st.session_state.chat.append({"role": "bot", "content": f"Error: {e}"})
    st.experimental_rerun()
