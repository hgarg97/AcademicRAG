import os
import streamlit as st
from RAG import AcademicRAG
from config import LLM_MODEL_OPTIONS
import config
from PIL import Image

retriever_mode = os.getenv("RAG_RETRIEVER_MODE", "faiss")
rag = AcademicRAG(retriever_mode=retriever_mode)

favicon_img = "images/favicon-96x96.png"

st.set_page_config(
    page_title="Academic RAG - Chatbot",
    layout="wide",
    page_icon = favicon_img
)

# st.markdown(
#     '<img src=favicon_img width="36" style="vertical-align: middle; margin-right: 10px;">'
#     '<span style="font-size: 28px; font-weight: bold;">Academic RAG - Chatbot</span>',
#     unsafe_allow_html=True
# )
col1, col2 = st.columns([1, 10])
with col1:
    st.image("images/favicon-96x96.png", width=60)
with col2:
    st.markdown("<div style='font-size: 40px; font-weight: bold;'>Academic RAG - Chatbot</div>", unsafe_allow_html=True)

st.markdown("""\
#### A research-driven platform transforming animal nutrition science through AI.

We leverage Retrieval-Augmented Generation (RAG) pipelines powered by LLMs like Meta’s LLaMA to deliver fast, context-aware answers from a curated library of scientific literature. This tool helps researchers, students, and industry professionals explore feed formulations, health outcomes, and advancements in livestock nutrition with structured, reference-backed insights.
""")


# === Sidebar: Global Settings ===
with st.sidebar:

    with open("images/logo.svg", "r") as f:
        svg_code = f.read()

    st.sidebar.markdown(
        f"<div style='width:50%; margin-bottom:2rem'>{svg_code}</div>",
        unsafe_allow_html=True
    )


    st.subheader("🎛️ Model Settings")

    available_display_names = list(LLM_MODEL_OPTIONS.keys())
    # Match default model from config
    default_display = next(name for name, internal in LLM_MODEL_OPTIONS.items() if internal == config.LLM_MODEL_NAME)
    selected_display = st.selectbox("Choose LLM Model:", available_display_names, index=available_display_names.index(default_display))

    selected_model = LLM_MODEL_OPTIONS[selected_display]

    st.markdown("---")

    # 🌡️ Temperature slider first
    temperature = st.slider("LLM Temperature (creativity)", min_value=0.0, max_value=1.5, value=0.7, step=0.1)

    # ✅ Now apply model/temperature settings
    if "selected_model" not in st.session_state or st.session_state.selected_model != selected_model:
        st.session_state.selected_model = selected_model
        rag.set_model(selected_model)

    rag.set_temperature(temperature)

    st.markdown(f"**🔧 Current model in use:** `{selected_model}`")

    st.markdown("---")

    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history = []
        st.session_state.upload_chat_history = []
        rag.cleanup_uploaded_pdfs()
        st.success("Conversation cleared and uploaded files deleted.")
        st.rerun()

# === Tabs: Chatbot | Summary | Upload ===
tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "📂 Upload PDFs", "📑 Paper Summary"])

# === Tab 1: Chatbot ===
with tab1:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(text)

    user_query = st.chat_input("Enter your research query...")
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        history_context = "".join([f"{r}: {t}\n" for r, t in st.session_state.chat_history[-4:]])

        with st.spinner("🔍 Retrieving academic context..."):
            chunks, refs, files = rag.find_related_chunks(user_query, top_k=10)
            response = rag.generate_answer(user_query, chunks, refs, history_context)

        with st.expander("🔍 Retrieved Context Chunks (Click to Expand)"):
            for idx, (chunk, file) in enumerate(chunks):
                st.markdown(f"**Chunk {idx+1}:**")
                st.info(chunk)

        with st.expander("📂 Source Papers (Click to Expand)"):
            for file in files:
                path = f"{rag.raw_pdf_dir}/{file}"
                st.markdown(f"[📄 {file}]({path})", unsafe_allow_html=True)

        with st.chat_message("assistant", avatar="🤖"):
            st.write(response)

        st.session_state.chat_history.append(("user", user_query))
        st.session_state.chat_history.append(("assistant", response))

# === Tab 2: Uploaded PDFs ===
with tab2:
    st.subheader("📂 Upload and Query Your PDFs")
    uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("📄 Chunking and embedding uploaded PDFs..."):
            for uploaded_file in uploaded_files:
                rag.process_uploaded_pdf(uploaded_file)
        st.success("✅ All PDFs processed!")

    if rag.user_pdf_chunks:
        uploaded_file_names = list({chunk["file_name"] for chunk in rag.user_pdf_chunks})
        selected_file = st.selectbox("Select a file to query:", uploaded_file_names)

        query = st.chat_input("Ask a question based on your uploaded PDF...")
        if query:
            with st.chat_message("user"):
                st.write(query)

            history_context = "".join([f"{r}: {t}\n" for r, t in st.session_state.get("upload_chat_history", [])[-4:]])

            with st.spinner("🔍 Retrieving from uploaded PDFs..."):
                chunks, refs, files, response = rag.query_uploaded_pdfs(query, history_context, selected_file)

            with st.expander("🔍 Retrieved Chunks (Click to Expand)"):
                for idx, (chunk, file) in enumerate(chunks):
                    st.markdown(f"**Chunk {idx+1} from {file}:**")
                    st.info(chunk)

            with st.chat_message("assistant", avatar="🤖"):
                st.write(response)

            if "upload_chat_history" not in st.session_state:
                st.session_state.upload_chat_history = []

            st.session_state.upload_chat_history.append(("user", query))
            st.session_state.upload_chat_history.append(("assistant", response))

# === Tab 3: Summary ===
with tab3:
    st.subheader("📑 Summarize a Research Paper")
    paper_titles = [""] + [title for title in rag.get_all_paper_titles() if title.lower() != "unknown title"]
    selected_paper = st.selectbox("Select a paper:", paper_titles, index=0, placeholder="Choose a paper to summarize")
    if not selected_paper:
        st.info("Please select a paper to generate the summary.")
    elif st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            summary = rag.summarize_paper(selected_paper)
            st.markdown(summary)
