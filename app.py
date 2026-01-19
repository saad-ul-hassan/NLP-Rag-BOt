import streamlit as st
import os
import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# 1. Configuration & Security
# Use Streamlit Secrets for the API Key in production
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
genai.configure(api_key=GEMINI_API_KEY)

# 2. Project Constants from your report
MODEL_NAME = "gemini-1.5-flash" 
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_DIR = "data/"

st.set_page_config(page_title="NLPAssist+", page_icon="🎓")
st.title("🎓 NLPAssist+ (CCP Project)")

# 3. Optimized Ingestion (Runs only once)
@st.cache_resource
def build_knowledge_base():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    documents, filenames = [], []
    
    if os.path.exists(DATA_DIR):
        for file in os.listdir(DATA_DIR):
            if file.endswith(".txt"):
                with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
                    # Paragraph-based splitting as per your report
                    chunks = f.read().split("\n\n")
                    for chunk in chunks:
                        if chunk.strip():
                            documents.append(chunk.strip())
                            filenames.append(file)
    
    if documents:
        embeddings = embed_model.encode(documents)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        return index, documents, filenames, embed_model
    return None, [], [], embed_model

index, docs, sources, embed_model = build_knowledge_base()

# 4. RAG Logic
def generate_rag_response(query):
    # Search top 3 relevant chunks
    query_vec = embed_model.encode([query])
    _, I = index.search(np.array(query_vec).astype('float32'), k=3)
    
    context = "\n".join([docs[i] for i in I[0] if i != -1])
    found_sources = list(set([sources[i] for i in I[0] if i != -1]))

    # Prompt Engineering
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer based ONLY on the context above:"
    
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text, found_sources

# 5. Simple Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask a university policy question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Retrieving information..."):
        answer, refs = generate_rag_response(prompt)
        full_reply = f"{answer}\n\n**Sources:** {', '.join(refs)}"
        st.chat_message("assistant").write(full_reply)
        st.session_state.messages.append({"role": "assistant", "content": full_reply})