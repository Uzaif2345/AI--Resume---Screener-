import os
import re
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import streamlit as st
from typing import List, Dict, Optional
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import requests
from transformers import AutoTokenizer, AutoModel, pipeline
import logging
from functools import lru_cache
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import base64
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.card import card
from streamlit_extras.let_it_rain import rain
import socket
import PyPDF2
from pdfminer.high_level import extract_text
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if running in Streamlit
def is_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except:
        return False

# Constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_WORKERS = 4  # For parallel processing
TIMEOUT = 60  # seconds
SUPPORTED_FILE_TYPES = ["pdf", "txt"]
COVER_LETTER_MODEL = "gpt2"  # Consider "gpt2-medium" or "gpt2-large" for better quality

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# --- Initialize NLP Models ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Enhanced skills dictionary
TECH_SKILLS = {
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'swift', 'kotlin',
    'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'sqlite', 'oracle',
    'html', 'css', 'sass', 'less', 'bootstrap', 'tailwind',
    'react', 'angular', 'vue', 'django', 'flask', 'fastapi', 'node.js', 'express',
    'git', 'github', 'gitlab', 'bitbucket', 'docker', 'kubernetes', 'jenkins', 'ansible',
    'aws', 'azure', 'gcp', 'terraform', 'serverless',
    'machine learning', 'deep learning', 'computer vision', 'nlp',
    'tensorflow', 'pytorch', 'keras', 'scikit-learn',
    'data analysis', 'data science', 'pandas', 'numpy', 'matplotlib', 'seaborn',
    'big data', 'spark', 'hadoop', 'hive', 'kafka',
    'tableau', 'powerbi', 'looker', 'excel'
}

# Request models
class ResumeRankRequest(BaseModel):
    job_desc: str
    files: List[UploadFile]

# --- Model Loading with Fallback and Caching ---
@lru_cache(maxsize=1)
def load_embedding_model():
    models_to_try = [
        'sentence-transformers/all-MiniLM-L6-v2',  # Smaller, faster model (default)
        'bert-base-uncased'                        # Fallback model
    ]
    
    for model_name in models_to_try:
        try:
            logger.info(f"Attempting to load model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            logger.info(f"Successfully loaded model: {model_name}")
            return tokenizer, model
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {str(e)}")
            continue
    
    raise RuntimeError("Could not load any embedding model")

@lru_cache(maxsize=1)
def load_cover_letter_generator():
    try:
        logger.info("Loading cover letter generator model")
        # Consider using larger models for better quality
        return pipeline("text-generation", model=COVER_LETTER_MODEL)
    except Exception as e:
        logger.error(f"Failed to load cover letter generator: {str(e)}")
        return None

tokenizer, embedding_model = load_embedding_model()

# --- FastAPI App ---
app = FastAPI(
    title="AI Resume Screener API",
    description="API for ranking resumes based on job description matching with cover letter generation",
    version="1.1.0"
)

# Add CORS middleware to allow all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Helper Functions with Enhanced Features ---
def clean_text(text: str) -> str:
    """Clean and normalize text"""
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_skills(text: str) -> List[str]:
    """Enhanced skill extraction with multi-word matching"""
    try:
        text = clean_text(text)
        tokens = word_tokenize(text)
        
        # Process tokens
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        
        # Find matches
        found_skills = set()
        
        # Single word matches
        for token in tokens:
            if token in TECH_SKILLS:
                found_skills.add(token)
        
        # Multi-word matches
        text = ' '.join(tokens)
        for skill in TECH_SKILLS:
            if ' ' in skill and skill in text:
                found_skills.add(skill)
        
        return sorted(found_skills)
    except Exception as e:
        logger.error(f"Error extracting skills: {str(e)}")
        return []

@lru_cache(maxsize=1000)
def get_embedding(text: str) -> np.ndarray:
    """Cached embedding generation"""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return np.zeros(384)  # Return zero vector if error occurs

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Safe cosine similarity calculation"""
    try:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return np.dot(a, b) / (a_norm * b_norm)
    except Exception:
        return 0.0

def extract_text_from_file(file: UploadFile) -> Optional[str]:
    """Extract text from PDF or text file with robust error handling"""
    try:
        # Read file content into memory
        file_content = file.file.read()
        
        # Reset file pointer for potential re-reading
        file.file.seek(0)
        
        # Handle text files
        if file.content_type == "text/plain":
            try:
                return file_content.decode("utf-8")
            except UnicodeDecodeError:
                logger.warning(f"Failed to decode text file {file.filename} as UTF-8")
                return None
        
        # Handle PDF files
        elif file.content_type == "application/pdf":
            try:
                # First try PyPDF2 (faster)
                pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
                text = "\n".join([page.extract_text() for page in pdf_reader.pages])
                
                # If PyPDF2 returns empty text, try pdfminer (slower but more robust)
                if not text.strip():
                    logger.info(f"PyPDF2 returned empty text, trying pdfminer for {file.filename}")
                    text = extract_text(BytesIO(file_content))
                
                return text if text.strip() else None
                
            except PyPDF2.PdfReadError as e:
                logger.error(f"PDF read error for {file.filename}: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"PDF processing failed for {file.filename}: {str(e)}")
                return None
        
        # Unsupported file type
        else:
            logger.warning(f"Unsupported file type: {file.content_type} for {file.filename}")
            return None
            
    except Exception as e:
        logger.error(f"General file processing error for {file.filename}: {str(e)}")
        return None

def generate_cover_letter(job_desc: str, resume_text: str, filename: str) -> str:
    """Generate a personalized cover letter using AI"""
    try:
        generator = load_cover_letter_generator()
        if not generator:
            return "Could not load cover letter generator"
        
        # Create a prompt for the generator
        prompt = f"""
        Generate a professional cover letter for the candidate whose resume is below,
        tailored specifically for the job description provided.
        
        Job Title: {job_desc.splitlines()[0][:100]}
        
        Job Description:
        {job_desc[:2000]}
        
        Resume Content:
        {resume_text[:2000]}
        
        The candidate's name is {os.path.splitext(filename)[0]}.
        Write a compelling cover letter that:
        - Addresses the hiring manager professionally
        - Highlights 3-5 most relevant skills from the resume
        - Shows enthusiasm for the specific position
        - Is concise (3-4 paragraphs)
        - Ends with a strong call to action
        
        Professional Cover Letter:
        """
        
        # Generate the cover letter
        result = generator(
            prompt,
            max_length=1024,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        
        generated_text = result[0]['generated_text'].replace(prompt, "").strip()
        
        # Clean up the generated text
        generated_text = re.sub(r'\n+', '\n', generated_text)  # Remove extra newlines
        generated_text = generated_text.split("Sincerely,")[0]  # Cut off after closing
        
        return generated_text + "\n\nSincerely,\n[Your Name]"
    except Exception as e:
        logger.error(f"Error generating cover letter: {str(e)}")
        return f"Error generating cover letter: {str(e)}"

def process_resume(resume: UploadFile, job_embed: np.ndarray) -> Optional[Dict]:
    """Process a single resume with comprehensive error handling"""
    try:
        # Validate file size
        if resume.size > MAX_FILE_SIZE:
            logger.warning(f"File {resume.filename} exceeds size limit")
            return None
        
        # Extract text from file
        text = extract_text_from_file(resume)
        if not text:
            logger.warning(f"Could not extract text from {resume.filename}")
            return None
        
        # Generate embedding and calculate similarity
        resume_embed = get_embedding(text)
        similarity = cosine_similarity(job_embed, resume_embed)
        
        return {
            "filename": resume.filename,
            "text": text,  # Store the text for cover letter generation
            "score": float(similarity),
            "skills": extract_skills(text)
        }
    except Exception as e:
        logger.error(f"Error processing {resume.filename}: {str(e)}")
        return None

# --- API Endpoints with Enhanced Features ---
@app.post("/api/extract_skills", response_model=Dict[str, List[str]])
async def api_extract_skills(file: UploadFile = File(...)):
    """Extract skills from a resume"""
    try:
        if file.size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large (max 5MB)")
            
        text = extract_text_from_file(file)
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from file")
            
        return {"skills": extract_skills(text)}
    except Exception as e:
        logger.error(f"API extract_skills error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rank_resumes", response_model=List[Dict])
async def api_rank_resumes(job_desc: str = Form(...), resumes: List[UploadFile] = File(...)):
    """Rank multiple resumes against a job description"""
    try:
        # Validate input
        if not job_desc.strip():
            raise HTTPException(status_code=400, detail="Job description cannot be empty")
            
        if not resumes:
            raise HTTPException(status_code=400, detail="No resumes provided")
        
        # Generate job embedding
        job_embed = get_embedding(job_desc)
        
        # Process resumes in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_resume, resume, job_embed) for resume in resumes]
            results = [future.result() for future in futures if future.result() is not None]
        
        if not results:
            raise HTTPException(status_code=400, detail="Could not process any resumes")
        
        return sorted(results, key=lambda x: x["score"], reverse=True)
    except Exception as e:
        logger.error(f"API rank_resumes error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Modern Streamlit UI with Enhanced Features ---
def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

def show_file_processing_error(uploaded_files, processed_files):
    """Display detailed error message about failed files"""
    failed_files = [f.name for f in uploaded_files if f.name not in [p["filename"] for p in processed_files]]
    
    with st.expander("⚠️ Processing Errors", expanded=True):
        st.error("Some files could not be processed. Common reasons:")
        
        reasons = [
            "File is password protected",
            "PDF contains scanned images (non-selectable text)",
            "Corrupted file",
            "Unsupported format (only PDF and TXT supported)"
        ]
        
        for reason in reasons:
            st.markdown(f"- {reason}")
        
        st.markdown("**Failed files:**")
        for file in failed_files:
            st.markdown(f"- `{file}`")

def show_network_info():
    """Display network information in a collapsible section"""
    local_ip = get_local_ip()
    with st.expander("🌐 Network Access Information", expanded=False):
        st.markdown(f"""
        - **Local Access:** [http://localhost:8501](http://localhost:8501)
        - **Network Access:** [http://{local_ip}:8501](http://{local_ip}:8501)
        - **API Endpoint:** [http://{local_ip}:8000](http://{local_ip}:8000)
        """)
        st.caption("Share these URLs with others on the same network")

def streamlit_ui():
    # Only run this if we're in Streamlit context
    if not is_streamlit():
        return
    
    # Page config
    st.set_page_config(
        page_title="AI Resume Screener Pro",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        /* Main container */
        .stApp {
            background-color: #f5f9ff;
        }
        
        /* Text areas */
        .stTextArea textarea {
            border-radius: 15px !important;
            padding: 15px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        /* Buttons */
        .stButton button {
            background: linear-gradient(135deg, #6B73FF 0%, #000DFF 100%) !important;
            color: white !important;
            border-radius: 12px !important;
            padding: 12px 28px !important;
            font-weight: bold !important;
            border: none !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease !important;
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        /* File uploader */
        .stFileUploader {
            border: 2px dashed #6B73FF !important;
            border-radius: 15px !important;
            padding: 25px !important;
            background-color: rgba(107, 115, 255, 0.05) !important;
        }
        
        /* Progress bar */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #6B73FF 0%, #000DFF 100%);
        }
        
        /* Cards */
        .st-expander {
            border-radius: 15px !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
        }
        
        /* Cover letter section */
        .cover-letter {
            background-color: #f8f9fa;
            border-left: 4px solid #6B73FF;
            padding: 15px;
            border-radius: 0 8px 8px 0;
            margin: 15px 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with gradient
    with stylable_container(
        key="header",
        css_styles="""
            {
                background: linear-gradient(135deg, #6B73FF 0%, #000DFF 100%);
                border-radius: 0 0 15px 15px;
                padding: 2rem;
                color: white;
                margin-bottom: 2rem;
            }
        """
    ):
        st.markdown("<h1 style='text-align: center; color: white;'>AI Resume Screener Pro 🔍</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: white; font-size: 1.2rem;'>Automatically rank candidates and generate cover letters</p>", unsafe_allow_html=True)
    
    # Network information (collapsed by default)
    show_network_info()
    
    # How it works cards
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            card(
                title="1. Upload Job Description",
                text="Paste your job requirements",
                image="https://cdn-icons-png.flaticon.com/512/2933/2933245.png",
                styles={
                    "card": {
                        "width": "100%",
                        "height": "300px",
                        "border-radius": "15px",
                        "box-shadow": "0 4px 8px rgba(0,0,0,0.1)"
                    },
                    "title": {
                        "color": "#2E86C1"
                    }
                }
            )
        
        with col2:
            card(
                title="2. Add Resumes",
                text="Upload multiple resumes (PDF/TXT)",
                image="https://cdn-icons-png.flaticon.com/512/942/942748.png",
                styles={
                    "card": {
                        "width": "100%",
                        "height": "300px",
                        "border-radius": "15px",
                        "box-shadow": "0 4px 8px rgba(0,0,0,0.1)"
                    },
                    "title": {
                        "color": "#2E86C1"
                    }
                }
            )
        
        with col3:
            card(
                title="3. Get Results + Cover Letters",
                text="Receive ranked candidates with AI-generated cover letters",
                image="https://cdn-icons-png.flaticon.com/512/190/190411.png",
                styles={
                    "card": {
                        "width": "100%",
                        "height": "300px",
                        "border-radius": "15px",
                        "box-shadow": "0 4px 8px rgba(0,0,0,0.1)"
                    },
                    "title": {
                        "color": "#2E86C1"
                    }
                }
            )
    
    # Main input section
    with st.container():
        st.markdown("---")
        col1, col2 = st.columns([3, 2])
        
        with col1:
            with stylable_container(
                key="job_desc_box",
                css_styles="""
                    {
                        border-radius: 15px;
                        padding: 20px;
                        background-color: white;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    }
                """
            ):
                st.subheader("📝 Job Description")
                job_desc = st.text_area(
                    "", 
                    height=200,
                    placeholder="Paste the job description here...",
                    label_visibility="collapsed"
                )
        
        with col2:
            with stylable_container(
                key="upload_box",
                css_styles="""
                    {
                        border-radius: 15px;
                        padding: 20px;
                        background-color: white;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    }
                """
            ):
                st.subheader("📂 Upload Resumes")
                uploaded_files = st.file_uploader(
                    "", 
                    type=SUPPORTED_FILE_TYPES, 
                    accept_multiple_files=True,
                    help=f"Max {MAX_FILE_SIZE//(1024*1024)}MB per file. Supported formats: {', '.join(SUPPORTED_FILE_TYPES)}",
                    label_visibility="collapsed"
                )
    
    # Process button
    st.markdown("""
    <div style="text-align: center; margin: 30px 0;">
    """, unsafe_allow_html=True)
    
    process_btn = st.button("✨ Rank Candidates & Generate Cover Letters", type="primary", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Results section
    if process_btn:
        if not job_desc:
            st.warning("Please enter a job description!", icon="⚠️")
        elif not uploaded_files:
            st.warning("Please upload at least one resume!", icon="⚠️")
        else:
            with st.spinner("Analyzing resumes and generating cover letters... This may take a few moments"):
                try:
                    # Prepare files for API call
                    files = [("resumes", file) for file in uploaded_files]
                    
                    # Call API with timeout
                    api_url = f"http://{get_local_ip()}:8000/api/rank_resumes"
                    response = requests.post(
                        api_url,
                        data={"job_desc": job_desc},
                        files=files,
                        timeout=TIMEOUT
                    )
                    
                    if response.status_code != 200:
                        error_detail = response.json().get('detail', 'Unknown error')
                        st.error(f"Error: {error_detail}")
                        
                        # Show specific error for file processing failures
                        if "Could not process any resumes" in error_detail:
                            show_file_processing_error(uploaded_files, [])
                        return
                    
                    results = response.json()
                    
                    if not results:
                        st.warning("No valid resumes found or processed", icon="⚠️")
                        show_file_processing_error(uploaded_files, [])
                        return
                    
                    # Check if some files failed to process
                    if len(results) < len(uploaded_files):
                        show_file_processing_error(uploaded_files, results)
                    
                    # Display results with celebration
                    rain(
                        emoji="🎉",
                        font_size=30,
                        falling_speed=5,
                        animation_length=1,
                    )
                    
                    st.success(f"Analysis complete! Processed {len(results)}/{len(uploaded_files)} resumes successfully.")
                    
                    # Top candidate highlight
                    if len(results) > 0:
                        top_candidate = results[0]
                        with stylable_container(
                            key="top_candidate",
                            css_styles="""
                                {
                                    border-radius: 15px;
                                    padding: 20px;
                                    background: linear-gradient(135deg, #6B73FF 0%, #000DFF 100%);
                                    color: white;
                                    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
                                    margin-bottom: 30px;
                                }
                            """
                        ):
                            st.markdown(f"### 🏆 Top Match: {top_candidate['filename']}")
                            st.markdown(f"**Match Score:** {top_candidate['score']:.2%}")
                            st.progress(top_candidate['score'])
                            
                            skills = ", ".join(top_candidate['skills'][:8])
                            st.markdown(f"**Key Skills:** {skills}")
                    
                    # All candidates list with cover letters
                    st.subheader("All Candidates with AI-Generated Cover Letters", divider="blue")
                    
                    for i, candidate in enumerate(results[:10], 1):
                        with st.expander(f"🎯 #{i}: {candidate['filename']} (Score: {candidate['score']:.2%})", expanded=(i==1)):
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                st.metric("Match Score", f"{candidate['score']:.2%}")
                                st.progress(candidate['score'])
                            
                            with col2:
                                st.markdown("**Key Skills:**")
                                skills_html = " ".join(
                                    [f"<span style='display: inline-block; background-color: #E1E8FF; color: #2E3B8F; padding: 4px 12px; margin: 4px; border-radius: 20px;'>{skill}</span>" 
                                     for skill in candidate['skills'][:10]]
                                )
                                st.markdown(skills_html, unsafe_allow_html=True)
                            
                            # Cover letter section
                            st.markdown("---")
                            st.subheader("📝 AI-Generated Cover Letter")
                            
                            with st.spinner(f"Generating personalized cover letter for {candidate['filename']}..."):
                                cover_letter = generate_cover_letter(job_desc, candidate['text'], candidate['filename'])
                            
                            # Display cover letter with nice formatting
                            with stylable_container(
                                key="cover_letter_box",
                                css_styles="""
                                    {
                                        border-radius: 8px;
                                        padding: 20px;
                                        background-color: #f8f9fa;
                                        border-left: 4px solid #6B73FF;
                                    }
                                """
                            ):
                                st.markdown(cover_letter)
                            
                            # Add download button for the cover letter
                            st.download_button(
                                label=f"📄 Download Cover Letter for {candidate['filename']}",
                                data=cover_letter,
                                file_name=f"cover_letter_{os.path.splitext(candidate['filename'])[0]}.txt",
                                mime="text/plain",
                                key=f"cover_letter_{i}"
                            )
                    
                    # Download all results as CSV
                    csv = pd.DataFrame(results).to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download All Results as CSV",
                        data=csv,
                        file_name="resume_ranking_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                except requests.exceptions.Timeout:
                    st.error("Processing took too long. Please try with fewer resumes.", icon="⏱️")
                except Exception as e:
                    st.error(f"Failed to process: {str(e)}", icon="❌")
                    logger.exception("Streamlit UI error")

# --- Run App ---
if __name__ == "__main__":
    import uvicorn
    import threading
    import webbrowser
    import time
    import sys
    
    # Check if we're in Streamlit context
    if not is_streamlit():
        # Get local IP address
        local_ip = get_local_ip()
        
        # Find available port for FastAPI
        def find_available_port(start_port):
            port = start_port
            while True:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('0.0.0.0', port))
                        return port
                except OSError:
                    port += 1
        
        fastapi_port = find_available_port(8000)
        
        # Start FastAPI in background thread
        fastapi_thread = threading.Thread(
            target=uvicorn.run,
            args=(app,),
            kwargs={
                "host": "0.0.0.0",
                "port": fastapi_port,
                "log_level": "info"
            },
            daemon=True
        )
        fastapi_thread.start()
        
        # Print access information
        print("\n" + "="*50)
        print(f"🔌 FastAPI Backend running at:")
        print(f"   - Local: http://localhost:{fastapi_port}")
        print(f"   - Network: http://{local_ip}:{fastapi_port}")
        print("\n" + "="*50)
        
        # Launch Streamlit
        print(f"🚀 Streamlit UI running at:")
        print(f"   - Local: http://localhost:8501")
        print(f"   - Network: http://{local_ip}:8501")
        print("="*50 + "\n")
        
        # Open browser automatically (only once)
        time.sleep(2)  # Wait for servers to start
        webbrowser.open_new_tab(f"http://{local_ip}:8501")
        
        # Run Streamlit with disabled auto-reload
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", __file__, "--server.runOnSave", "false"]
        sys.exit(stcli.main())
    else:
        # This is a Streamlit execution context
        streamlit_ui()