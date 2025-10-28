"""Streamlit dashboard for Visual Data Explorer."""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="Visual Data Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🔍 Visual Data Explorer</div>', unsafe_allow_html=True)
st.markdown('---')

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data source selection
    data_source = st.selectbox(
        "Select Data Source",
        ["Upload Files", "Sample Dataset", "Web Scraping"]
    )
    
    # Visualization options
    st.subheader("Visualization Settings")
    show_clusters = st.checkbox("Show Clusters", value=True)
    show_embeddings = st.checkbox("Show Embeddings", value=False)
    
    # Clustering parameters
    st.subheader("Clustering Parameters")
    n_clusters = st.slider("Number of Clusters", 2, 10, 5)
    clustering_method = st.selectbox(
        "Clustering Method",
        ["KMeans", "DBSCAN", "Hierarchical"]
    )

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🖼️ Images", "📝 Text", "🔎 Search"])

with tab1:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Total Items", value="0", delta="+0")
    with col2:
        st.metric(label="Images", value="0", delta="+0")
    with col3:
        st.metric(label="Text Documents", value="0", delta="+0")
    with col4:
        st.metric(label="Tables", value="0", delta="+0")
    
    st.markdown('<div class="section-header">Cluster Distribution</div>', unsafe_allow_html=True)
    
    # Sample cluster visualization
    if show_clusters:
        # Generate sample data for demonstration
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
            'cluster': np.random.randint(0, n_clusters, 100)
        })
        
        fig = px.scatter(
            sample_data, 
            x='x', 
            y='y', 
            color='cluster',
            title='Sample Cluster Visualization',
            labels={'cluster': 'Cluster ID'}
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown('<div class="section-header">Image Gallery</div>', unsafe_allow_html=True)
    
    # File upload for images
    uploaded_images = st.file_uploader(
        "Upload Images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    if uploaded_images:
        cols = st.columns(3)
        for idx, uploaded_file in enumerate(uploaded_images):
            with cols[idx % 3]:
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, use_container_width=True)
    else:
        st.info("Upload images to see them here!")

with tab3:
    st.markdown('<div class="section-header">Text Documents</div>', unsafe_allow_html=True)
    
    # File upload for text
    uploaded_text = st.file_uploader(
        "Upload Text Files",
        type=['txt', 'csv'],
        accept_multiple_files=True
    )
    
    if uploaded_text:
        for uploaded_file in uploaded_text:
            with st.expander(uploaded_file.name):
                if uploaded_file.type == 'text/csv':
                    df = pd.read_csv(uploaded_file)
                    st.dataframe(df)
                else:
                    content = uploaded_file.read().decode('utf-8')
                    st.text_area("Content", content, height=200)
    else:
        st.info("Upload text files or CSVs to see them here!")

with tab4:
    st.markdown('<div class="section-header">Search & Explore</div>', unsafe_allow_html=True)
    
    search_query = st.text_input("🔎 Search Query", placeholder="Enter search terms...")
    
    col1, col2 = st.columns(2)
    with col1:
        search_type = st.radio(
            "Search Type",
            ["Text", "Image Similarity", "Semantic Search"]
        )
    with col2:
        top_k = st.number_input("Number of Results", min_value=1, max_value=50, value=10)
    
    if st.button("🔍 Search"):
        if search_query:
            st.success(f"Searching for: '{search_query}'...")
            st.info("Search functionality will be implemented with actual data.")
        else:
            st.warning("Please enter a search query.")

# Footer
st.markdown('---')
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Visual Data Explorer | AI/ML Project for Heterogeneous Data Visualization</p>
    <p>Built with Streamlit 🎈</p>
</div>
""", unsafe_allow_html=True)
