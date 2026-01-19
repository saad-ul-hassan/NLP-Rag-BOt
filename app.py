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

# Project Constants
MODEL_NAME = "gemini-1.5-flash" # [cite: 27]
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # [cite: 28]
# CHANGE: Ab folder ki bajaye current directory check hogi
DATA_DIR = "./" 

st.set_page_config(page_title="NLPAssist+", page_icon="🎓")
st.title("🎓 NLPAssist+ (CCP Project)")

# --- 2. INGESTION ENGINE (Paragraph-based splitting) [cite: 33, 36] ---
@st.cache_resource
def build_knowledge_base():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    documents, filenames = [], []
    
    # Current directory mein files dhoondna 
    for file in os.listdir(DATA_DIR):
        # Sirf .txt files read karna aur code files ko ignore karna
        if file.endswith(".txt") and file != "requirements.txt":
            with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
                # Double newline split strategy as per your report [cite: 36]
                chunks = f.read().split("\n\n")
                for chunk in chunks:
                    if chunk.strip():
                        documents.append(chunk.strip())
                        filenames.append(file)
    
    # Create FAISS Index [cite: 29, 41]
    if documents:
        embeddings = embed_model.encode(documents)
        dimension = embeddings.shape[1] # 384 dimensions [cite: 41]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        return index, documents, filenames, embed_model
    return None, [], [], embed_model

# Initialize the system [cite: 59, 60]
index, docs, sources, embed_model = build_knowledge_base()

# --- 3. RAG PIPELINE LOGIC [cite: 18, 23] ---
def generate_rag_response(query):
    # CRITICAL FIX: Empty index check
    if index is None:
        return "⚠️ Knowledge base empty! Please ensure your .txt files are uploaded to GitHub (Main Directory).", []
    
    # 1. Retrieval: Vectorize query and search FAISS [cite: 44, 45]
    query_vec = embed_model.encode([query])
    _, I = index.search(np.array(query_vec).astype('float32'), k=3) # Top-3 chunks [cite: 45]
    
    context_chunks = []
    found_sources = []
    for i in I[0]:
        if i != -1:
            context_chunks.append(docs[i])
            found_sources.append(sources[i])
    
    context_text = "\n\n".join(context_chunks)
    unique_sources = list(set(found_sources))

    # 2. Generation: Construct Prompt for Gemini [cite: 46, 49]
    prompt = f"""You are a helpful university assistant called NLPAssist+. 
    Use the following context to answer the student's question accurately.
    If the answer is not in the context, politely say you don't know. [cite: 65]

    Context:
    {context_text}

    Question: {query}
    Answer:"""
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        # Parameters: top_p=0.9, temp=0.7 as per CCP report [cite: 50]
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
        return f"Error: {str(e)}", []

# --- 4. CHAT UI [cite: 31, 61] ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about university policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Retrieving Answer..."):
        answer, refs = generate_rag_response(prompt)
        
        # Format with sources like your demo [cite: 52, 61]
        if refs:
            full_reply = f"{answer}\n\n**Sources:** {', '.join(refs)}"
        else:
            full_reply = answer
            
        with st.chat_message("assistant"):
            st.markdown(full_reply)
        st.session_state.messages.append({"role": "assistant", "content": full_reply})