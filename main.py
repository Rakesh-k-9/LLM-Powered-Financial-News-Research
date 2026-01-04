import os
import time
import streamlit as st
# IMPORTANT: block TensorFlow
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# ---------------- UI ----------------
st.set_page_config(page_title="EquiLens", layout="wide")
st.title("🧠 EquiLens: LLM-Powered Financial News Research")
st.sidebar.title("🔗 News Article URLs")
st.sidebar.markdown("Paste financial news URLs and build a knowledge base.")

urls = []
for i in range(3):
    url = st.sidebar.text_input(
        f"News URL {i+1}",
        placeholder="https://www.moneycontrol.com/..."
    )
    if url:
        urls.append(url)

process_btn = st.sidebar.button("🚀 Process URLs")

status = st.empty()
FAISS_DIR = "faiss_index"
# ---------------- LLM SETUP ----------------
hf_pipeline = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=128
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)
# ---------------- PROCESS URLS ----------------
if process_btn and urls:
    status.info("🔄 Loading articles...")
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    # Attach source URLs explicitly
    for doc, url in zip(data, urls):
        doc.metadata["source"] = url

    status.info("✂️ Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30
    )
    docs = text_splitter.split_documents(data)

    status.info("🧠 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(docs, embeddings)

    status.info("💾 Saving vector index...")
    vectorstore.save_local(FAISS_DIR)

    status.success("✅ Knowledge base created successfully!")

st.markdown("---")
st.subheader("❓ Ask a Question")

query = st.text_input(
    "Enter your financial research question",
    placeholder="Ask a question about the financial news"
)

if query:
    if os.path.exists(FAISS_DIR):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.load_local(
            FAISS_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=retriever,
            reduce_k_below_max_tokens=True,
            max_tokens_limit=450
        )

        with st.spinner("🔍 Analyzing news..."):
            result = chain({"question": query})

        st.subheader("📌 Answer")
        st.write(result["answer"])

        if result.get("sources"):
            st.subheader("🔗 Sources")
            for src in result["sources"].split("\n"):
                st.write(src)
    else:
        st.warning("⚠️ Please process URLs first.")


