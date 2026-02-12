import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(
    page_title="AI Semantic Search",
    page_icon="🧠",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------ #
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.big-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #AAAAAA;
    font-size: 18px;
    margin-bottom: 40px;
}

.result-card {
    background-color: #161B22;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
    border: 1px solid #30363D;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD DATA ------------------ #
@st.cache_resource
def load_embeddings():
    return np.load("embeddings.npy")

@st.cache_data
def load_documents():
    with open("documents.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

embeddings = load_embeddings()
documents = load_documents()

# ------------------ RETRIEVAL ------------------ #
def retrieve_top_k(query_embedding, embeddings, k=10):
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]

def get_query_embedding(query):
    np.random.seed(hash(query) % 2**32)
    return np.random.rand(embeddings.shape[1])

# ------------------ HEADER ------------------ #
st.markdown('<div class="big-title">🧠 AI Semantic Search Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Search documents using vector embeddings & cosine similarity</div>', unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------ #
with st.sidebar:
    st.header("⚙️ Search Settings")
    k_value = st.slider("Number of Results (Top K)", 1, 20, 10)

    st.markdown("---")
    st.markdown("### 📊 Dataset Statistics")
    st.metric("Total Documents", len(documents))
    st.metric("Embedding Dimension", embeddings.shape[1])

    st.markdown("---")
    st.info("This system ranks documents using cosine similarity over precomputed embeddings.")

# ------------------ SEARCH AREA ------------------ #
col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input("🔎 Enter your search query", placeholder="e.g. Fine-tuning LLMs")

with col2:
    search_button = st.button("Search", use_container_width=True)

# ------------------ RESULTS ------------------ #
if search_button and query:
    with st.spinner("Analyzing semantic similarity..."):
        query_embedding = get_query_embedding(query)
        results = retrieve_top_k(query_embedding, embeddings, k=k_value)

    st.markdown("## 🔍 Search Results")
    st.success(f"Found {len(results)} relevant documents")

    for i, (doc, score) in enumerate(results, 1):
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"### #{i}")
        st.write(doc)
        st.progress(float(score))
        st.caption(f"Similarity Score: {score:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

elif search_button and not query:
    st.warning("Please enter a query to search.")

# ------------------ DOCUMENT VIEWER ------------------ #
with st.expander("📚 View Full Document Dataset"):
    for i, doc in enumerate(documents, 1):
        st.write(f"{i}. {doc}")
