import streamlit as st
import os
import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# --- 1. CONFIGURATION & SECURITY ---
# Streamlit Cloud ke Secrets se API Key uthane ke liye 
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

# Project Constants [cite: 27, 28, 35]
MODEL_NAME = "gemini-1.5-flash" 
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_DIR = "data/"

st.set_page_config(page_title="NLPAssist+", page_icon="🎓")
st.title("🎓 NLPAssist+ (CCP Project)")

# --- 2. INGESTION ENGINE ---
@st.cache_resource
def build_knowledge_base():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    documents, filenames = [], []
    
    # Check if data directory exists [cite: 35]
    if os.path.exists(DATA_DIR):
        for file in os.listdir(DATA_DIR):
            if file.endswith(".txt"):
                with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
                    # Splitting by double newlines as per your CCP report [cite: 36]
                    chunks = f.read().split("\n\n")
                    for chunk in chunks:
                        if chunk.strip():
                            documents.append(chunk.strip())
                            filenames.append(file)
    
    # Create FAISS Index only if documents are found [cite: 41, 59]
    if documents:
        embeddings = embed_model.encode(documents)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        return index, documents, filenames, embed_model
    return None, [], [], embed_model

# Initialize the system [cite: 59, 69]
index, docs, sources, embed_model = build_knowledge_base()

# --- 3. RAG PIPELINE LOGIC ---
def generate_rag_response(query):
    # CRITICAL FIX: Check if index exists to prevent AttributeError [cite: 21, 45]
    if index is None:
        return "⚠️ Knowledge base empty! Please ensure the 'data/' folder on GitHub contains .txt files.", []
    
    # 1. Retrieval: Search top 3 relevant chunks [cite: 45]
    query_vec = embed_model.encode([query])
    _, I = index.search(np.array(query_vec).astype('float32'), k=3)
    
    context_chunks = []
    found_sources = []
    for i in I[0]:
        if i != -1:
            context_chunks.append(docs[i])
            found_sources.append(sources[i])
    
    context_text = "\n\n".join(context_chunks)
    unique_sources = list(set(found_sources))

    # 2. Generation: Construct Prompt and call Gemini [cite: 46, 49]
    prompt = f"""You are a helpful university assistant called NLPAssist+. 
    Use the following context to answer the student's question accurately.
    If the answer is not in the context, politely say you don't know.

    Context:
    {context_text}

    Question: {query}
    Answer:"""
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        # Parameters from your report: top_p=0.9, temp=0.7 [cite: 50]
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                top_k=40
            )
        )
        return response.text, unique_sources
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}", []

# --- 4. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history [cite: 61]
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input [cite: 21]
if prompt := st.chat_input("Ask about university policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Analyzing documents..."):
        answer, refs = generate_rag_response(prompt)
        
        # Format response with citations as per your screenshots [cite: 61, 52]
        if refs:
            full_reply = f"{answer}\n\n**Sources:** {', '.join(refs)}"
        else:
            full_reply = answer
            
        with st.chat_message("assistant"):
            st.markdown(full_reply)
        st.session_state.messages.append({"role": "assistant", "content": full_reply})