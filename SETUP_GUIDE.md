# Job Recommendation System - Setup Guide

## Overview
This system scrapes LinkedIn job postings and uses AI embeddings to match your resume with the best job opportunities.

## Prerequisites
- Python 3.8+
- pip (Python package manager)

## Setup Instructions

### Step 1: Install Required Dependencies
```bash
pip install streamlit pandas numpy sentence-transformers scikit-learn requests beautifulsoup4 PyPDF2 pinecone-client
```

### Step 2: Run the Data Collection Notebook
1. Open `Linkedin_Scraper_Tutorial.ipynb` in Jupyter Notebook or VS Code
2. Run all cells to:
   - Scrape LinkedIn job postings for "SDE", "Python Developer", "Data Engineer" in India
   - Create job embeddings
   - Save `india.csv`, `job_embeddings.npy`, and `job_data.json`

### Step 3: Run the Streamlit App
Navigate to the directory containing `app.py` and run:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Features
- 📤 **Upload Resume**: Support for PDF and TXT formats
- 🎯 **Job Matching**: AI-powered semantic matching using embeddings
- 📊 **Match Scores**: Similarity percentage for each job
- 🔗 **Direct Links**: Quick access to LinkedIn job postings
- 📥 **Export Results**: Download recommendations as CSV
- ⚙️ **Customizable**: Adjust number of recommendations (3-20)

## Usage
1. Prepare your resume (PDF or TXT format)
2. Open the Streamlit app
3. Upload your resume using the file uploader
4. View personalized job recommendations sorted by match score
5. Click on LinkedIn links to view full job descriptions
6. Download results as CSV for future reference

## File Structure
```
Downloads/
├── Linkedin_Scraper_Tutorial.ipynb  # Data collection notebook
├── app.py                           # Streamlit application
├── india.csv                        # Job data (auto-generated)
├── job_embeddings.npy              # Job embeddings (auto-generated)
├── job_data.json                   # Job metadata (auto-generated)
└── your_resume.pdf                 # Your resume file
```

## Troubleshooting

### "Job data not found" Error
- Make sure you've run the Linkedin_Scraper_Tutorial.ipynb notebook completely
- Check that `job_embeddings.npy` and `job_data.json` exist in the same directory as `app.py`

### PDF Reading Issues
- Ensure your PDF is not password-protected
- Try converting to a text file (TXT) if PDF doesn't work

### Model Loading is Slow
- The first time you run the app, it downloads the embedding model (~60MB)
- This is cached, so subsequent runs will be faster

## Job Search Parameters
Current settings search for:
- **Job Titles**: SDE, Python Developer, Data Engineer
- **Location**: India
- **Results**: 100+ job listings

To modify these, edit the `Linkedin_Scraper_Tutorial.ipynb` notebook and update the parameters.

## Performance Tips
- Use a TXT version of your resume for faster processing
- The app caches the model and data for faster subsequent loads
- Adjust the number of recommendations slider to balance between speed and results

## License
This tool is for educational and personal use only.
