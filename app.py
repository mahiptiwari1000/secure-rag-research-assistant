import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
 
st.title("AI Security Research Assistant")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)

    text = ""
    for page in reader.pages:
        text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])

    st.write(f"Total chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./db")

    st.success("VectorDB created successfully")

    llm = ChatOpenAI(model="gpt-4o-mini")

    query = st.text_input("Ask a question about the PDF")

    if query:
        docs = vectordb.similarity_search(query, k=3)

        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
You are a helpful AI research assistant.

Use ONLY the context below to answer the question.

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""
        
        response = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)