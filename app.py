import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

st.title("Document Q&A")

# Load models once (cached so it doesn't reload every interaction)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
question = st.text_input("Ask a question about the document")

if uploaded_file and question:
    with st.spinner("Reading document..."):
        # Save uploaded file temporarily and extract text
        reader = PdfReader(uploaded_file)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

        # Chunk it
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=80)
        chunks = splitter.split_text(full_text)

        # Embed and store in a fresh in-memory Chroma collection
        embeddings = model.encode(chunks)
        chroma_client = chromadb.Client()  # in-memory, resets each run
        collection = chroma_client.get_or_create_collection(name="uploaded_doc")
        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )

    with st.spinner("Finding relevant sections..."):
        question_embedding = model.encode([question])
        results = collection.query(query_embeddings=question_embedding.tolist(), n_results=3)
        retrieved_chunks = results["documents"][0]

    with st.spinner("Generating answer..."):
        context = "\n\n---\n\n".join(retrieved_chunks)
        prompt = f"""Answer the question using ONLY the context below.
If the answer is not contained in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
        response = gemini_client.models.generate_content(model="gemini-3.6-flash", contents=prompt)

    st.write("### Answer")
    st.write(response.text)

    st.write("### Sources used")
    for i, chunk in enumerate(retrieved_chunks):
        with st.expander(f"Source {i+1}"):
            st.write(chunk)