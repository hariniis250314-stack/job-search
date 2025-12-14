import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import tempfile

import PyPDF2

# Set page config
st.set_page_config(
    page_title="Job Recommendation System",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
    <style>
    .job-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #1f77b4;
    }
    .match-score {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    .job-title {
        font-size: 20px;
        font-weight: bold;
        color: #0f1419;
    }
    .company-name {
        font-size: 16px;
        color: #65676b;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("💼 LinkedIn Job Recommendation System")
st.markdown("### Upload your resume and find the best job matches from LinkedIn!")

# Sidebar for settings
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Number of job recommendations", 3, 20, 10)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

# Function to extract text from text file
def extract_text_from_txt(text_file):
    """Extract text from text file"""
    try:
        return text_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading text file: {str(e)}")
        return None

# Function to match resume with jobs
@st.cache_resource
def load_model_and_data():
    """Load the embedding model and job data"""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None
    
    # Get the directory where the script is running
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_path = os.path.join(script_dir, 'job_embeddings.npy')
    data_path = os.path.join(script_dir, 'job_data.json')
    
    # Load job embeddings
    if os.path.exists(embeddings_path):
        job_embeddings = np.load(embeddings_path)
    else:
        st.error(f"Job embeddings not found at {embeddings_path}")
        return None, None, None
    
    # Load job metadata
    if os.path.exists(data_path):
        with open(data_path, 'r') as f:
            job_data_list = json.load(f)
    else:
        st.error(f"Job data not found at {data_path}")
        return None, None, None
    
    return model, job_embeddings, job_data_list

def find_best_jobs(resume_text, model, job_embeddings, job_data_list, top_k=10):
    """Find the best matching jobs for a given resume"""
    
    if resume_text is None or len(resume_text.strip()) == 0:
        st.error("Could not extract text from resume")
        return None
    
    # Generate embedding for resume
    resume_embedding = model.encode(resume_text)
    
    # Calculate similarity scores
    similarities = cosine_similarity([resume_embedding], job_embeddings)[0]
    
    # Get top K matches
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(top_indices, 1):
        job_data = job_data_list[int(idx)]
        results.append({
            'rank': rank,
            'job_title': job_data['job_title'],
            'company_name': job_data['company_name'],
            'job_link': job_data['job_link'],
            'similarity_score': float(similarities[idx]),
            'time_posted': job_data['time_posted'],
            'num_applicants': job_data['num_applicants']
        })
    
    return results

# Main app logic
def main():
    # Get the directory where the script is running
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if embeddings exist
    embeddings_path = os.path.join(script_dir, 'job_embeddings.npy')
    data_path = os.path.join(script_dir, 'job_data.json')
    
    if not os.path.exists(embeddings_path) or not os.path.exists(data_path):
        st.error("❌ Job data not found!")
        st.warning(f"Looking for files in: {script_dir}")
        st.info("Steps to set up:")
        st.markdown("""
        1. Run the Linkedin_Scraper_Tutorial notebook to scrape jobs
        2. This will create `india.csv`, `job_embeddings.npy`, and `job_data.json` files
        3. Make sure these files are in the same directory as this app
        4. Come back to this app after setup
        """)
        return
    
    # Load model and data
    with st.spinner("Loading model and job data..."):
        model, job_embeddings, job_data_list = load_model_and_data()
    
    if model is None:
        return
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📤 Upload Your Resume")
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "Choose a resume file",
            type=["pdf", "txt"],
            help="Supported formats: PDF, TXT"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    with col2:
        if uploaded_file is not None:
            st.subheader("🎯 Job Recommendations")
            st.markdown("---")
            
            # Extract text from resume
            with st.spinner("Processing resume..."):
                if uploaded_file.type == "application/pdf":
                    resume_text = extract_text_from_pdf(uploaded_file)
                else:
                    resume_text = extract_text_from_txt(uploaded_file)
            
            if resume_text:
                # Find matching jobs
                with st.spinner("Finding matching jobs..."):
                    matching_jobs = find_best_jobs(resume_text, model, job_embeddings, job_data_list, top_k)
                
                if matching_jobs:
                    # Display results
                    st.success(f"Found {len(matching_jobs)} matching jobs!")
                    
                    # Metrics at top
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Total Matches", len(matching_jobs))
                    with col_m2:
                        avg_score = np.mean([job['similarity_score'] for job in matching_jobs])
                        st.metric("Avg Match Score", f"{avg_score:.1%}")
                    with col_m3:
                        best_score = matching_jobs[0]['similarity_score']
                        st.metric("Best Match", f"{best_score:.1%}")
                    
                    st.markdown("---")
                    
                    # Display each job
                    for job in matching_jobs:
                        with st.container():
                            col_rank, col_content = st.columns([0.15, 0.85])
                            
                            with col_rank:
                                st.markdown(f"### #{job['rank']}")
                                score_percentage = f"{job['similarity_score']:.1%}"
                                st.markdown(f"<div class='match-score'>{score_percentage}</div>", unsafe_allow_html=True)
                            
                            with col_content:
                                st.markdown(f"<div class='job-title'>{job['job_title']}</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='company-name'>{job['company_name']}</div>", unsafe_allow_html=True)
                                
                                col_info1, col_info2, col_info3 = st.columns(3)
                                with col_info1:
                                    st.caption(f"📅 Posted: {job['time_posted']}")
                                with col_info2:
                                    st.caption(f"👥 Applicants: {job['num_applicants']}")
                                with col_info3:
                                    st.markdown(f"[🔗 View on LinkedIn]({job['job_link']})")
                            
                            st.markdown("---")
                    
                    # Export results
                    st.subheader("📥 Export Results")
                    
                    # Create DataFrame for export
                    export_df = pd.DataFrame(matching_jobs)
                    
                    # Download as CSV
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name="job_recommendations.csv",
                        mime="text/csv"
                    )
        else:
            st.info("👈 Upload a resume to get started!")

if __name__ == "__main__":
    main()
