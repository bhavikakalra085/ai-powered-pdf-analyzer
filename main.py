import os
import streamlit as st
import fitz
import chromadb
from openai import OpenAI

st.title("AI-Powered RAG PDF Analyzer")

api_key = st.text_input("OpenAI API Key", type="password")
pdf = st.file_uploader("Upload PDF", type="pdf")
question = st.text_input("Ask a question about your PDF")

if pdf and question and api_key:
    client = OpenAI(api_key=api_key)

    # 1. Extract PDF text
    doc = fitz.open(stream=pdf.getvalue(), filetype="pdf")
    chunks = []

    for page_no, page in enumerate(doc):
        text = page.get_text()
        for i in range(0, len(text), 1000):
            chunks.append({
                "text": text[i:i + 1000],
                "page": page_no + 1
            })

    if not chunks:
        st.error("No readable text found.")
        st.stop()

    # 2. Generate embeddings
    embeddings = client.embeddings.create(
        model="text-embedding-3-small",
        input=[c["text"] for c in chunks]
    )

    # 3. Store chunks in ChromaDB
    db = chromadb.Client()
    collection = db.create_collection(
        name="pdf_chunks",
        embedding_function=None
    )

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        embeddings=[e.embedding for e in embeddings.data],
        metadatas=[{"page": c["page"]} for c in chunks]
    )

    # 4. Embed the question and retrieve relevant chunks
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(4, len(chunks))
    )

    context = "\n\n".join(
        f"[Page {m['page']}]\n{t}"
        for t, m in zip(
            results["documents"][0],
            results["metadatas"][0]
        )
    )

    # 5. Generate an answer using the retrieved context
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the supplied PDF context. "
                    "If the answer is missing, say so. "
                    "Use the provided page labels for citations.\n"
                    + context
                )
            },
            {"role": "user", "content": question}
        ],
        temperature=0
    )

    st.subheader("Answer")
    st.write(response.choices[0].message.content)
tHis is my project ka code can you tell me what is HappeninG and wHat questions can be aksed in interview i know basic rag pipelione and what raG is
