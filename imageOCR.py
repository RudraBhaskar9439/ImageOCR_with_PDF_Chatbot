import streamlit as st
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
import io
from typing import List, Dict
import tempfile

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable")

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Classes and utility functions
class ChunkWithSource:
    def __init__(self, text: str, source: str):
        self.text = text
        self.source = source

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def process_image_with_gemini(image_path: str) -> str:
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        prompt = "Extract and return all text visible in this image. Format it clearly."
        response = model.generate_content([
            {"mime_type": f'image/{image_path.split(".")[-1].lower()}', "data": image_bytes},
            prompt
        ])
        return response.text
    except Exception as e:
        st.error(f"Error processing image with Gemini: {str(e)}")
        return ""

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    return splitter.split_text(text)

def sentence_encode(sentences):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model.encode(sentences)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Streamlit UI
st.set_page_config(page_title="DocQA with Gemini", layout="wide")
st.title("📄 Intelligent Document & Image QA with Gemini")

uploaded_files = st.file_uploader("📤 Upload PDFs and Images", type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Please upload at least one PDF or image file to proceed.")
    st.stop()

chunks_with_sources = []
temp_dir = tempfile.mkdtemp()

for uploaded_file in uploaded_files:
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.name.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
        chunks = split_text_into_chunks(text)
        chunks_with_sources.extend([ChunkWithSource(chunk, uploaded_file.name) for chunk in chunks])
    else:
        text = process_image_with_gemini(file_path)
        if text:
            chunks = split_text_into_chunks(text)
            chunks_with_sources.extend([ChunkWithSource(chunk, uploaded_file.name) for chunk in chunks])

all_chunks = [c.text for c in chunks_with_sources]
chunk_vectors = sentence_encode(all_chunks)

query = st.text_area("🧠 Ask a question about your documents:")
if query:
    query_vector = sentence_encode([query])[0]
    sims = [(cosine_similarity(vec, query_vector), i) for i, vec in enumerate(chunk_vectors)]
    top_chunks = sorted(sims, reverse=True)[:3]

    context = "\n".join([all_chunks[i] for _, i in top_chunks])
    sources = set([chunks_with_sources[i].source for _, i in top_chunks])

    prompt = f"""You are a helpful assistant.

Context:
{context}

Question: {query}

Provide a helpful and concise answer. Mention the source file(s): {', '.join(sources)}."""

    try:
        response = model.generate_content(prompt)
        st.markdown("### 🤖 Answer")
        st.write(response.text)
        st.markdown("---")
        st.markdown(f"**Sources:** {', '.join(sources)}")
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
