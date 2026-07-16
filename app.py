import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to path to ensure backend modules can be imported
sys.path.append(str(Path(__file__).resolve().parent))

# Load keys from Streamlit secrets if available (for deployment safety)
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass  # No secrets.toml found — keys will be loaded from .env instead

# Import backend modules
from backend.indexer import clone_repository, index_repository
from backend.retriever import get_indexed_repositories, get_chat_engine

# --- Page Configurations ---
st.set_page_config(
    page_title="CodeMind - AI Codebase Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Premium CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600&display=swap');

/* Main container styling */
html, body, [class*="css"] {
    font-family: 'Outfit', 'Inter', sans-serif;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background: #0d091e !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Titles and Headers */
.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #b088f9 0%, #da82eb 50%, #f770a1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
}
.subtitle {
    color: #8c889f;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.2rem;
    font-weight: 600;
    color: #b088f9;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}

/* Status messaging styling */
.success-box {
    padding: 1rem;
    background-color: rgba(40, 167, 69, 0.1);
    border-left: 4px solid #28a745;
    border-radius: 4px;
    color: #e2fcd5;
}
.error-box {
    padding: 1rem;
    background-color: rgba(220, 53, 69, 0.1);
    border-left: 4px solid #dc3545;
    border-radius: 4px;
    color: #fcd5d5;
}
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown("<div class='main-title'>🧠 CodeMind</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI Codebase Assistant — Clone, Index, and Chat with any repository</div>", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("<div class='section-header'>📥 Ingest Codebase</div>", unsafe_allow_html=True)
    repo_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/user/repo")
    
    # Check if API Key is configured
    api_key_check = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key_check:
        st.warning("⚠️ No API Key found in .env. Falling back to local offline embeddings and mock LLM.")
        
    if st.button("Clone & Index Repository", use_container_width=True):
        if not repo_url:
            st.error("Please enter a valid GitHub URL.")
        else:
            with st.spinner("Cloning repository and running ingestion pipeline..."):
                try:
                    repo_path = clone_repository(repo_url)
                    index_repository(repo_path)
                    st.toast("Success: Repository successfully indexed!", icon="🚀")
                    st.rerun() # Refresh list of indexed repos
                except Exception as e:
                    st.error(f"Error: {e}")
                    
    st.markdown("---")
    st.markdown("<div class='section-header'>📂 Repository Selection</div>", unsafe_allow_html=True)
    
    indexed_repos = get_indexed_repositories()
    if indexed_repos:
        selected_repo = st.selectbox(
            "Select an indexed codebase",
            options=indexed_repos,
            index=0
        )
    else:
        st.info("No repositories indexed yet. Enter a GitHub URL to start.")
        selected_repo = None

# --- Main Chat Area ---
if not selected_repo:
    st.info("👈 Please enter a GitHub repository URL and click **Clone & Index** (or select a codebase if already indexed) in the sidebar to get started.")
else:
    st.markdown(f"### Currently Chatting with: `{selected_repo}`")
    
    # Re-initialize chat engine when selected repo changes
    if "current_repo" not in st.session_state or st.session_state.current_repo != selected_repo:
        st.session_state.current_repo = selected_repo
        with st.spinner("Loading codebase index into memory..."):
            try:
                st.session_state.chat_engine = get_chat_engine(selected_repo)
                st.session_state.messages = []
            except Exception as e:
                st.error(f"Failed to load engine for `{selected_repo}`: {e}")
                
    # Initialize message list
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # If sources are stored, display them
            if message.get("sources"):
                with st.expander("📚 View source references"):
                    for idx, src in enumerate(message["sources"]):
                        st.markdown(f"**[{idx+1}] File:** `{src['file']}`" + (f" · `{src['language']}`" if src.get('language') else "") + f" (Relevance: {src['score']:.2f})")
                        st.code(src["code"], language="python")

    # Handle User Input
    if prompt := st.chat_input(f"Ask me anything about {selected_repo}..."):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display assistant response placeholder
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            # Using streaming chat response for premium feel
            try:
                chat_engine = st.session_state.chat_engine
                response_stream = chat_engine.stream_chat(prompt)
                
                # Stream the response generator
                # In LlamaIndex, the stream response is collected under response_stream.response_gen
                # We can output it character by character or chunk by chunk
                full_response = ""
                for chunk in response_stream.response_gen:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                
                # Fetch sources
                sources = []
                if hasattr(response_stream, "source_nodes") and response_stream.source_nodes:
                    for source_node in response_stream.source_nodes:
                        sources.append({
                            "file": source_node.node.metadata.get("file_path", "Unknown File"),
                            "language": source_node.node.metadata.get("language", ""),
                            "score": source_node.score if source_node.score is not None else 0.0,
                            "code": source_node.node.get_content()[:600] + "\n..."
                        })
                
                # Display sources expandable if any references found
                if sources:
                    with st.expander("📚 View source references"):
                        for idx, src in enumerate(sources):
                            st.markdown(f"**[{idx+1}] File:** `{src['file']}`" + (f" · `{src['language']}`" if src.get('language') else "") + f" (Relevance: {src['score']:.2f})")
                            st.code(src["code"], language="python")
                            
                # Save assistant response to session history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"An error occurred while communicating with the model: {e}")
