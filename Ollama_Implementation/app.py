import os
import streamlit as st
from RAG import AcademicRAG
from config import LLM_MODEL_OPTIONS
import config

retriever_mode = os.getenv("RAG_RETRIEVER_MODE", "faiss")
rag = AcademicRAG(retriever_mode=retriever_mode)


st.set_page_config(page_title="📘 Academic RAG - Chatbot", layout="wide")
st.title("📘 Academic RAG - Chatbot")
st.markdown("### Ask questions based on indexed research papers or your uploaded documents!")

# === Sidebar: Upload + Summarize ===
with st.sidebar:

    st.subheader("🎛️ Model Settings")

    available_display_names = list(LLM_MODEL_OPTIONS.keys())
    # Match default model from config
    default_display = next(name for name, internal in LLM_MODEL_OPTIONS.items() if internal == config.LLM_MODEL_NAME)
    selected_display = st.selectbox("Choose LLM Model:", available_display_names, index=available_display_names.index(default_display))

    selected_model = LLM_MODEL_OPTIONS[selected_display]

    # 🌡️ Temperature slider first
    temperature = st.slider("LLM Temperature (creativity)", min_value=0.0, max_value=1.5, value=0.7, step=0.1)

    # ✅ Now apply model/temperature settings
    if "selected_model" not in st.session_state or st.session_state.selected_model != selected_model:
        st.session_state.selected_model = selected_model
        rag.set_model(selected_model)

    rag.set_temperature(temperature)

    st.markdown(f"**🔧 Current model in use:** `{selected_model}`")


    st.markdown("---")

    st.header("📄 Upload Your PDF")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        rag.process_uploaded_pdf(uploaded_file)
        st.success("✅ PDF uploaded and chunked!")

    st.markdown("---")
    st.subheader("📑 Summarize a Research Paper")
    paper_titles = rag.get_all_paper_titles()
    selected_paper = st.selectbox("Select a paper:", paper_titles)
    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            summary = rag.summarize_paper(selected_paper)
            st.markdown(summary)

# === Chat History ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(text)

# === Chat Input ===
user_query = st.chat_input("Enter your research query...")
if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    history_context = "".join([f"{r}: {t}\n" for r, t in st.session_state.chat_history[-4:]])

    source = st.sidebar.radio("Select Query Source:", ["Academic Dataset", "Uploaded PDF"], key="source_selector")
    with st.spinner("Retrieving relevant information..."):
        if source == "Uploaded PDF" and rag.user_pdf_chunks:
            chunks, refs, files, response = rag.query_uploaded_pdfs(user_query, history_context)
        else:
            chunks, refs, files = rag.find_related_chunks(user_query, top_k=10)
            response = rag.generate_answer(user_query, chunks, refs, history_context)

    # === Retrieved Chunks ===
    with st.expander("🔍 Retrieved Context Chunks (Click to Expand)"):
        for idx, (chunk, file) in enumerate(chunks):
            st.markdown(f"**Chunk {idx+1}:**")
            st.info(chunk)

    # === Source Papers ===
    with st.expander("📂 Source Papers (Click to Expand)"):
        for file in files:
            path = f"{rag.raw_pdf_dir}/{file}"
            st.markdown(f"[📄 {file}]({path})", unsafe_allow_html=True)

    # === Assistant Response ===
    with st.chat_message("assistant", avatar="🤖"):
        st.write(response)

    # === Update Session State ===
    st.session_state.chat_history.append(("user", user_query))
    st.session_state.chat_history.append(("assistant", response))

# === Reset Button ===
if st.button("🗑️ Clear Conversation"):
    st.session_state.chat_history = []
    rag.cleanup_uploaded_pdfs()
    st.success("Conversation cleared and temporary files deleted.")
    st.rerun()
