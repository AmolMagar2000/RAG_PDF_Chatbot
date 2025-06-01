import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# 2) Patch torch to avoid Streamlit watcher errors
import torch
torch.classes.__path__ = []


import os
import subprocess
import streamlit as st
import asyncio
import time

# ── App configuration ──────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Chat Expert", layout="wide")

# Fix for Windows event loop issue
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ── PDF processing dependencies ────────────────────────────────────────────────
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader

# ── Paths & CLI helper ─────────────────────────────────────────────────────────
MODEL_PATH = os.path.abspath("models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf")
CLI_PATH   = os.path.abspath("bin/bitnet-cli")

st.write(f"Loading model from: `{MODEL_PATH}`")
st.write(f"Exists? {os.path.exists(MODEL_PATH)}")

def generate_with_bitnet_stream(
    prompt: str,
    threads: int = 4,
    n_predict: int = 128,
    temp: float = 0.2,
    top_p: float = 0.9,
):
    """
    Launch the BitNet CLI and yield each character as soon as it's emitted.
    """
    cmd = [
        CLI_PATH,
        "--model", MODEL_PATH,
        "--prompt", prompt,
        "--threads", str(threads),
        "--n_predict", str(n_predict),
        "--temp", str(temp),
        "--top_p", str(top_p),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=1,
        text=True,
    )
    while True:
        char = proc.stdout.read(1)
        if not char:
            break
        yield char
    proc.wait()

# ── Session state initialization ───────────────────────────────────────────────
st.session_state.setdefault('chat_history', [])
st.session_state.setdefault('vector_store', None)
st.session_state.setdefault('all_docs', [])

# ── App header ────────────────────────────────────────────────────────────────
st.title("🤖 PDF Chat Expert")
st.write("Upload and initialize your PDF knowledge base, then ask expert-level questions.")

# ── PDF database setup ─────────────────────────────────────────────────────────
if st.button("Initialize PDF Database"):
    with st.spinner("Processing PDFs and creating vector store..."):
        def load_and_clean_docs():
            pdf_paths = [
                os.path.join(root, f)
                for root, _, files in os.walk('rag-dataset')
                for f in files if f.lower().endswith('.pdf')
            ]
            if not pdf_paths:
                st.error("No PDF files found in 'rag-dataset' directory!")
                return []

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            docs = []
            for path in pdf_paths:
                loader = PyMuPDFLoader(path)
                pages = loader.load_and_split(splitter)
                for page in pages:
                    page.metadata['source'] = os.path.basename(path)
                docs.extend(pages)
            return [d for d in docs if len(d.page_content.strip()) > 100]

        def create_store(docs):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            return FAISS.from_documents(docs, embeddings)

        all_docs = load_and_clean_docs()
        if all_docs:
            st.session_state.all_docs = all_docs
            st.session_state.vector_store = create_store(all_docs)
            st.success(f"Loaded {len(all_docs)} document chunks.")

# ── User input ─────────────────────────────────────────────────────────────────
user_input = st.text_area("Your question:", height=150)
submit = st.button("Submit")

# ── Query handler ──────────────────────────────────────────────────────────────
def handle_query(question: str) -> str:
    if not st.session_state.vector_store:
        st.error("Please initialize the PDF database first.")
        return ""

    retriever = st.session_state.vector_store.as_retriever(search_kwargs={'k': 2})
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return "I’m sorry, but I couldn’t find any relevant information."

    context = "\n\n".join(
        f"Source ({doc.metadata.get('source','unknown')}):\n{doc.page_content}"
        for doc in docs
    )
    prompt = (
        "You are an assistant trained to answer questions **strictly based on the provided context**.Do not add any external knowledge. "
        f"\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )

    placeholder = st.empty()
    output = ""

    for ch in generate_with_bitnet_stream(prompt, threads=4, n_predict=128):
        output += ch
        placeholder.markdown(output)

    return output

# ── Run on Submit ──────────────────────────────────────────────────────────────
if user_input and submit:
    with st.spinner("Generating answer..."):
        start = time.time()
        ans = handle_query(user_input)
        elapsed = time.time() - start
        st.session_state.chat_history.append((user_input, ans, elapsed))

# ── Chat history display ───────────────────────────────────────────────────────
st.divider()
st.subheader("Chat History")
for q, a, t in reversed(st.session_state.chat_history):
    with st.expander(f"Q: {q}", expanded=True):
        st.markdown(f"**Answer:**\n{a}")
        st.markdown(f"**Response Time:** {t:.2f} seconds")
        st.divider()
