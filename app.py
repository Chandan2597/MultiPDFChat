import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from groq import Groq

# ----------------------------
# 🔑 SET YOUR GROQ API KEY
# ----------------------------
GROQ_API_KEY = "GROQ_API_KEY_HERE"
client = Groq(api_key=GROQ_API_KEY)

# ----------------------------
# 📄 Extract text from PDFs
# ----------------------------
def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# ----------------------------
# ✂️ Split text into chunks
# ----------------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

# ----------------------------
# 🧠 Create FAISS Vector Store
# ----------------------------
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# ----------------------------
# 🤖 Groq LLM Call
# ----------------------------
def ask_groq(context, question):
    prompt = f"""
    Answer the question based only on the provided context.
    If the answer is not in the context, say:
    "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # fast & free on Groq
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content

# ----------------------------
# 💬 Chat with PDF
# ----------------------------
def chat_with_pdf(question):
    if not os.path.exists("faiss_index"):
        return "⚠️ Please upload and process PDFs first."

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    return ask_groq(context, question)

# ----------------------------
# 🌐 Streamlit UI
# ----------------------------
st.set_page_config(page_title="Chat with PDFs", layout="wide")

st.title("💬 Chat with Multiple PDFs (RAG + Groq)")

# Sidebar
with st.sidebar:
    st.header("📄 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Process PDFs"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            raw_text = get_pdf_text(uploaded_files)
            chunks = get_text_chunks(raw_text)

            get_vector_store(chunks)

            st.success("✅ PDFs processed successfully!")

# Chat state
if "history" not in st.session_state:
    st.session_state.history = []

# Chat UI
st.header("Ask a Question")
question = st.text_input("Type your question here")

if st.button("Submit"):
    if not question:
        st.warning("Please enter a question.")
    else:
        answer = chat_with_pdf(question)
        st.session_state.history.append((question, answer))

# Show chat history
for q, a in reversed(st.session_state.history):
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")