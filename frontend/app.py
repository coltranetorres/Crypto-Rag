"""
Streamlit Frontend for RAG Application
"""
import streamlit as st
import requests
from typing import Optional
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .context-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 0.75rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

def check_api_health() -> bool:
    """Check if API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def query_rag(question: str, top_k: int = 5, use_agent: bool = False) -> Optional[dict]:
    """Query the RAG API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "question": question,
                "top_k": top_k,
                "use_agent": use_agent
            },
            timeout=120  # 2 minute timeout for long queries
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying API: {str(e)}")
        return None

def get_stats() -> Optional[dict]:
    """Get API statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        response.raise_for_status()
        return response.json()
    except:
        return None

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_healthy" not in st.session_state:
    st.session_state.api_healthy = False

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # API Health Check
    st.subheader("API Status")
    if st.button("Check API Health"):
        st.session_state.api_healthy = check_api_health()
    
    if st.session_state.api_healthy:
        st.success("✅ API is healthy")
    else:
        st.error("❌ API is not responding")
        st.info("Make sure the FastAPI backend is running:\n```bash\nuvicorn backend.main:app --reload\n```")
    
    # Query Settings
    st.subheader("Query Settings")
    top_k = st.slider("Number of chunks to retrieve", 1, 10, 5)
    use_agent = st.checkbox("Use Agent-based query (slower but more intelligent)", value=False)
    
    # Statistics
    st.subheader("📊 Statistics")
    if st.button("Refresh Stats"):
        stats = get_stats()
        if stats:
            st.json(stats)
    
    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main content
st.markdown('<div class="main-header">🤖 RAG Chat Assistant</div>', unsafe_allow_html=True)
st.markdown("Ask questions about the documents in the knowledge base!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show metadata if available
        if message.get("metadata"):
            with st.expander("📊 Query Metrics"):
                metadata = message["metadata"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Latency", f"{metadata.get('total_latency_ms', 0):.2f} ms")
                with col2:
                    st.metric("Retrieval Latency", f"{metadata.get('retrieval_latency_ms', 0):.2f} ms")
                with col3:
                    st.metric("Generation Latency", f"{metadata.get('generation_latency_ms', 0):.2f} ms")
                
                if metadata.get("usage"):
                    st.json(metadata["usage"])
        
        # Show contexts if available
        if message.get("contexts"):
            with st.expander(f"📚 Sources ({len(message['contexts'])} chunks)"):
                for i, ctx in enumerate(message["contexts"], 1):
                    st.markdown(f"**Source {i}:** {ctx['source']}")
                    st.markdown(f"*Similarity: {ctx['similarity']:.3f}*")
                    if ctx.get("page_numbers"):
                        st.markdown(f"*Pages: {', '.join(map(str, ctx['page_numbers']))}*")
                    st.markdown(f"```\n{ctx['text'][:200]}...\n```")
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Query API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = query_rag(prompt, top_k=top_k, use_agent=use_agent)
        
        if result:
            # Display answer
            st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
            
            # Display metrics
            with st.expander("📊 Query Metrics"):
                metadata = result.get("metadata", {})
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Latency", f"{metadata.get('total_latency_ms', 0):.2f} ms")
                with col2:
                    st.metric("Retrieval", f"{metadata.get('retrieval_latency_ms', 0):.2f} ms")
                with col3:
                    st.metric("Generation", f"{metadata.get('generation_latency_ms', 0):.2f} ms")
                with col4:
                    st.metric("Chunks Retrieved", metadata.get('chunks_retrieved', 0))
                
                # Token usage
                if metadata.get("usage"):
                    st.markdown("**Token Usage:**")
                    st.json(metadata["usage"])
            
            # Display contexts
            if result.get("contexts"):
                with st.expander(f"📚 Sources ({len(result['contexts'])} chunks retrieved)"):
                    for i, ctx in enumerate(result["contexts"], 1):
                        st.markdown(f'<div class="context-box">', unsafe_allow_html=True)
                        st.markdown(f"**Source {i}:** `{ctx['source']}`")
                        st.markdown(f"*Similarity Score: {ctx['similarity']:.3f}*")
                        if ctx.get("page_numbers"):
                            st.markdown(f"*Page Numbers: {', '.join(map(str, ctx['page_numbers']))}*")
                        st.markdown("**Context:**")
                        st.text_area(
                            f"Context {i}",
                            ctx['text'],
                            height=150,
                            key=f"context_{i}",
                            label_visibility="collapsed"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                        if i < len(result["contexts"]):
                            st.divider()
            
            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "metadata": result.get("metadata"),
                "contexts": result.get("contexts")
            })
        else:
            st.error("Failed to get response from API. Please check the backend connection.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>RAG Chat Assistant | Powered by Strands + OpenRouter</p>
    </div>
    """,
    unsafe_allow_html=True
)

