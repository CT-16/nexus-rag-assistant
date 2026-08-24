import streamlit as st
import os
import time
from dotenv import load_dotenv
from document_loader import DocumentManager
from rag_engine import RAGEngine

# Load environment variables
load_dotenv()

# Page Setup
st.set_page_config(page_title="NexusRAG — AI Knowledge Suite", page_icon="⚡", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .stChatInput { margin-top: 20px; }
    .source-box { background-color: #0F172A; border-radius: 8px; padding: 12px; margin-top: 8px; }
    </style>
""", unsafe_allow_html=True)

# Session States Initialization
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "is_indexed" not in st.session_state:
    st.session_state.is_indexed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "welcome_sent" not in st.session_state:
    st.session_state.welcome_sent = False

# --- SIDEBAR: CONTROL & KNOWLEDGE SETUP ---
with st.sidebar:
    st.title("⚡ NexusRAG Control")
    st.caption("Enterprise Document Processing Engine")
    
    if st.session_state.is_indexed:
        st.success("✅ Document Knowledge Base Active")
    else:
        st.info("👋 Upload document(s) below to initialize AI memory.")

    # Secure API Key Handling
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = st.text_input("🔑 Gemini API Key:", type="password", help="Kept private in session state.")
        
    st.divider()
    
    st.subheader("📁 1. Knowledge Base Upload")
    uploaded_files = st.file_uploader("Upload PDF Documents:", type=["pdf"], accept_multiple_files=True)
    
    with st.expander("⚙️ Knowledge Retrieval Tuning"):
        st.markdown("**Passage Chunk Size**")
        st.caption("Higher values keep larger sections of text together.")
        chunk_size = st.slider("Characters per block:", 800, 2500, 1200, 100)
        
        st.markdown("**Context Overlap**")
        st.caption("Ensures sentence context isn't sliced at boundaries.")
        chunk_overlap = st.slider("Overlapping characters:", 100, 500, 250, 50)
        
        st.markdown("**Retrieval Depth (Top K)**")
        st.caption("Higher depth ensures multi-page architectures aren't missed.")
        top_k = st.slider("Retrieved document sections:", 2, 12, 6)

    if st.button("🚀 Process Documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.error("⚠️ Please upload at least one PDF file.")
        elif not api_key:
            st.error("⚠️ Gemini API key missing.")
        else:
            with st.status("🧠 Synthesizing & Indexing Knowledge...", expanded=True) as status:
                st.write("📂 Extracting structured text across pages...")
                time.sleep(0.4)
                
                try:
                    if st.session_state.rag_engine is None:
                        st.session_state.rag_engine = RAGEngine(api_key=api_key)
                    
                    st.write("✂️ Generating high-density contextual chunks...")
                    loader = DocumentManager(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    chunks = loader.process_uploaded_files(uploaded_files)
                    
                    st.write("🧩 Indexing vector embeddings into FAISS memory...")
                    total_indexed = st.session_state.rag_engine.build_vector_store(chunks)
                    
                    st.session_state.is_indexed = True
                    st.session_state.top_k = top_k
                    st.session_state.welcome_sent = False
                    
                    status.update(label=f"✅ Memory Ready! Indexed {total_indexed} sections.", state="complete", expanded=False)
                    st.rerun()
                except Exception as e:
                    status.update(label="❌ Ingestion failed.", state="error")
                    st.error(f"Error: {e}")

    if st.session_state.is_indexed:
        st.divider()
        if st.button("🗑️ Reset & Clear Knowledge Base", type="secondary", use_container_width=True):
            st.session_state.is_indexed = False
            st.session_state.chat_history = []
            st.session_state.welcome_sent = False
            st.session_state.rag_engine = None
            st.rerun()

# --- MAIN INTERFACE: INTELLIGENT CHAT EXPERIENCE ---
st.markdown("<h1 style='text-align: center;'>⚡ NexusRAG AI Knowledge Suite</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94A3B8;'>Grounded Multi-Document Analysis & Technical Intelligence</p>", unsafe_allow_html=True)
st.divider()

# --- STATE 1: UNINITIALIZED (WELCOME ONBOARDING) ---
if not st.session_state.is_indexed:
    col1, col2, col3 = st.columns([1, 2.2, 1])
    with col2:
        st.markdown("""
        ### 👋 Welcome to NexusRAG!
        
        Get started by loading your project files, security reports, or technical specifications:
        
        1. **Upload your PDF files** using the left sidebar.
        2. Click **🚀 Process Documents** to index your files.
        3. Ask questions in any format (deep paragraphs, comparative tables, executive summaries).
        """)

# --- STATE 2: ACTIVE KNOWLEDGE BASE ---
else:
    # Initial Greeting after upload
    if not st.session_state.welcome_sent:
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": "✨ **Knowledge Base Active!** I have completely processed and indexed your uploaded documents.\n\nYou can now ask me for detailed technical explanations, system architecture overviews (in paragraphs, lists, or tables), payload analyses, or key findings.",
            "follow_up": "Would you like an executive summary or a full architectural breakdown of the uploaded documents?",
            "sources": []
        })
        st.session_state.welcome_sent = True

    # Render Conversation Log
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Highlighted Follow-Up Callout Box
            if msg.get("follow_up"):
                st.info(f"💡 **Suggested Next Query:** {msg['follow_up']}")
            
            # Collapsible Cited Sources
            if msg.get("sources"):
                with st.expander("📌 View Verified PDF Citations"):
                    for idx, doc in enumerate(msg["sources"], 1):
                        src_name = doc.metadata.get("source_file", "Unknown File")
                        page_num = doc.metadata.get("page", 0) + 1
                        st.markdown(f"**Source {idx}:** `{src_name}` — **Page {page_num}**")
                        st.caption(doc.page_content)
                        st.divider()

    # User Query Input
    user_query = st.chat_input("Ask a question (e.g., 'Explain full system architecture in multi-paragraph format')...")
    
    if user_query:
        # Render User Question
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # Synthesize AI Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document context & synthesizing response..."):
                try:
                    k_val = getattr(st.session_state, 'top_k', 6)
                    result = st.session_state.rag_engine.query(user_query, top_k=k_val)
                    
                    # Display Answer
                    st.markdown(result["answer"])
                    
                    # Display Highlighted Follow-Up Box
                    if result["follow_up"]:
                        st.info(f"💡 **Suggested Next Query:** {result['follow_up']}")
                    
                    # Display Citations
                    if result["sources"]:
                        with st.expander("📌 View Verified PDF Citations"):
                            for idx, doc in enumerate(result["sources"], 1):
                                src_name = doc.metadata.get("source_file", "Unknown File")
                                page_num = doc.metadata.get("page", 0) + 1
                                st.markdown(f"**Source {idx}:** `{src_name}` — **Page {page_num}**")
                                st.caption(doc.page_content)
                                st.divider()
                            
                    # Save to History
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": result["answer"],
                        "follow_up": result["follow_up"],
                        "sources": result["sources"]
                    })
                except Exception as e:
                    st.error(f"Execution Error: {e}")