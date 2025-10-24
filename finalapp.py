import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import plotly.graph_objects as go
import plotly.express as px
from skin_disease_model import SkinDiseaseDetector
import warnings
warnings.filterwarnings('ignore')
import requests
import shutil
import time
from streamlit.components.v1 import html
import base64

MODEL_URL = "https://www.dropbox.com/scl/fi/5wwmx63gw24afr15hxhid/skin_disease_model.h5?rlkey=we18mf6adx26eeh6hkmqss4a6&st=c1o0j81k&dl=1"
MODEL_PATH = "skin_disease_model.h5"
TEMP_MODEL_PATH = "skin_disease_model.h5?dl=1"

def download_model():
    if not os.path.exists(MODEL_PATH):
        try:
            st.info("Downloading model from Dropbox...")
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(TEMP_MODEL_PATH, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            # Rename if needed
            if os.path.exists(TEMP_MODEL_PATH):
                shutil.move(TEMP_MODEL_PATH, MODEL_PATH)
            st.success("Model downloaded and ready.")
        except Exception as e:
            st.error(f"Failed to download model: {e}")

download_model()

if not os.path.exists(MODEL_PATH):
    st.error("Model file was not downloaded. Please check the Dropbox link or network connection.")
else:
    st.success("Model file is present.")

# Set page config
st.set_page_config(
    page_title="DermoraSense - Skin Disease Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS with cursor-reactive background and layered design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #0a0a0a 25%, #0f0f23 50%, #0a0a0a 75%, #000000 100%);
        color: #ffffff;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        overflow-x: hidden;
        position: relative;
        min-height: 100vh;
    }
    
    .main .block-container {
        background: transparent;
        padding: 0;
        max-width: 100%;
        position: relative;
        z-index: 1;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Background */
    .formless-background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -1;
        background: linear-gradient(135deg, #000000 0%, #0a0a0a 25%, #0f0f23 50%, #0a0a0a 75%, #000000 100%);
        overflow: hidden;
    }
    
    /* Dynamic Background Overlay */
    .background-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -1;
        background: radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.05) 30%, transparent 70%);
        animation: backgroundPulse 8s ease-in-out infinite;
    }
    
    /* Enhanced cursor-reactive background */
    .cursor-reactive-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -1;
        pointer-events: none;
        background: radial-gradient(circle at var(--mouse-x, 50%) var(--mouse-y, 50%), 
                                  rgba(99, 102, 241, 0.15) 0%, 
                                  rgba(139, 92, 246, 0.1) 20%, 
                                  rgba(168, 85, 247, 0.05) 40%, 
                                  transparent 60%);
        transition: all 0.1s ease-out;
    }
    
    @keyframes backgroundPulse {
        0%, 100% {
            opacity: 0.3;
            transform: scale(1);
        }
        50% {
            opacity: 0.6;
            transform: scale(1.1);
        }
    }
    
    /* Cursor-Reactive Particles */
    .particle {
        position: absolute;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.8) 0%, rgba(139, 92, 246, 0.4) 50%, transparent 100%);
        border-radius: 50%;
        pointer-events: none;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        animation: float 8s ease-in-out infinite;
        filter: blur(0.5px);
    }
    
    .particle:nth-child(odd) {
        animation-delay: 0s;
        animation-duration: 10s;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.8) 0%, rgba(99, 102, 241, 0.4) 50%, transparent 100%);
    }
    
    .particle:nth-child(even) {
        animation-delay: 2s;
        animation-duration: 12s;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.8) 0%, rgba(168, 85, 247, 0.4) 50%, transparent 100%);
    }
    
    .particle:nth-child(3n) {
        animation-delay: 4s;
        animation-duration: 14s;
        background: radial-gradient(circle, rgba(168, 85, 247, 0.8) 0%, rgba(99, 102, 241, 0.4) 50%, transparent 100%);
    }
    
    @keyframes float {
        0%, 100% {
            transform: translateY(0px) translateX(0px) scale(1) rotate(0deg);
            opacity: 0.4;
        }
        25% {
            transform: translateY(-30px) translateX(15px) scale(1.2) rotate(90deg);
            opacity: 0.7;
        }
        50% {
            transform: translateY(-50px) translateX(-10px) scale(0.8) rotate(180deg);
            opacity: 0.5;
        }
        75% {
            transform: translateY(-25px) translateX(-20px) scale(1.1) rotate(270deg);
            opacity: 0.8;
        }
    }
    
    /* Enhanced cursor trail effect */
    .cursor-trail {
        position: fixed;
        width: 6px;
        height: 6px;
        background: radial-gradient(circle, rgba(99, 102, 241, 1) 0%, rgba(139, 92, 246, 0.8) 50%, rgba(168, 85, 247, 0.4) 100%);
        border-radius: 50%;
        pointer-events: none;
        z-index: 9999;
        transition: all 0.15s ease;
        box-shadow: 0 0 30px rgba(99, 102, 241, 0.8), 0 0 60px rgba(139, 92, 246, 0.4);
        filter: blur(0.5px);
    }
    
    /* Style Streamlit file uploader */
    .stFileUploader {
        background: rgba(170, 187, 230, 1);
        border: 2px dashed rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(20px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
        max-width: 720px; /* constrain width */
        margin-left: auto;
        margin-right: auto; /* center */
    }
    
    .stFileUploader:hover {
        border-color: rgba(99, 102, 241, 0.6);
        transform: translateY(-2px);
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.1);
    }
    
    /* Additional styling for file uploader elements */
    .stFileUploader > div { color: #456ab5 !important; }
    
    /* Force high-contrast placeholder text inside the dropzone */
    .stFileUploader label,
    .stFileUploader p,
    .stFileUploader span,
    .stFileUploader small,
    .stFileUploader strong,
    .stFileUploader div[data-testid="stFileUploader"] {
        color: #2d3748 !important;
        opacity: 1 !important;
        font-weight: 500 !important;
    }
    
    /* Specific styling for the drag and drop text */
    .stFileUploader div[data-testid="stFileUploader"] p,
    .stFileUploader div[data-testid="stFileUploader"] span,
    .stFileUploader div[data-testid="stFileUploader"] label {
        color: #2d3748 !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        opacity: 1 !important;
    }
    
    /* Improve cloud icon visibility */
    .stFileUploader svg {
        fill: #6366f1 !important;
        stroke: #EBEDF5!important;
        opacity: 0.99 !important;
        filter: drop-shadow(0 0 6px rgba(99,102,241,0.35));
    }
    
    .stFileUploader button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.875rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.3) !important;
    }
    
    /* Make sure the uploaded file chip is visible */
    .stFileUploader .uploadedFile {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
    }

    /* Override any Streamlit default styling that might hide text */
    .stFileUploader * {
        color: #2d3748 !important;
        opacity: 1 !important;
    }
    
    /* Specific override for the main upload text */
    .stFileUploader div[data-testid="stFileUploader"] > div > div > div {
        color: #2d3748 !important;
        font-weight: 500 !important;
    }
    
    /* Additional styling to ensure text visibility */
    .stFileUploader label[data-testid="stFileUploader"] {
        color: #2d3748 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        text-align: center !important;
        display: block !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Style the file uploader container */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        text-align: center !important;
    }
    
    /* Ensure all text elements are visible */
    .stFileUploader p, .stFileUploader span, .stFileUploader div {
        color: #2d3748 !important;
        font-weight: 500 !important;
        opacity: 1 !important;
    }
    
    /* Custom file upload button */
    .custom-upload-btn {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        cursor: pointer;
        border: none;
        display: inline-block;
        text-decoration: none;
        margin-top: 1rem;
    }
    
    .custom-upload-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.3);
        text-decoration: none;
        color: white;
    }
    
    /* Hero Section with dynamic background */
    .hero-section {
        background: transparent;
        min-height: 80vh;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        overflow: hidden;
        z-index: 1;
        padding: 2rem;
        margin-bottom: 1.5rem; /* tighten space before next section */
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.1) 30%, transparent 70%);
        pointer-events: none;
        animation: pulse 6s ease-in-out infinite;
        z-index: -1;
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 0.3;
            transform: scale(1);
        }
        50% {
            opacity: 0.6;
            transform: scale(1.2);
        }
    }
    
    .hero-content {
        text-align: center;
        max-width: 95%;
        padding: 0 1rem;
        z-index: 2;
        position: relative;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem 1.5rem;
        backdrop-filter: blur(20px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        margin: 0 auto;
    }
    
    .hero-content:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateY(-5px);
        box-shadow: 0 35px 70px rgba(0, 0, 0, 0.4);
    }
    
    .hero-title {
        font-size: clamp(2.5rem, 6vw, 4rem);
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #6366f1 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1.5rem;
        letter-spacing: -0.02em;
        line-height: 1.1;
        text-shadow: 0 0 30px rgba(99, 102, 241, 0.3);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .hero-subtitle {
        font-size: clamp(1rem, 2.5vw, 1.5rem);
        font-weight: 600;
        color: #e5e7eb;
        margin-bottom: 1rem;
        opacity: 0.9;
        line-height: 1.3;
    }
    
    .hero-description {
        font-size: clamp(0.9rem, 2vw, 1rem);
        color: #9ca3af;
        line-height: 1.5;
        margin-bottom: 2rem;
        max-width: 90%;
        margin-left: auto;
        margin-right: auto;
        padding: 0 1rem;
    }
    
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 3rem;
        flex-wrap: wrap;
    }
    
    .stat-item {
        text-align: center;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stat-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.1), transparent);
        transition: left 0.5s ease;
    }
    
    .stat-item:hover::before {
        left: 100%;
    }
    
    .stat-item:hover {
        transform: translateY(-4px);
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.1);
    }
    
    .stat-number {
        display: block;
        font-size: 2.5rem;
        font-weight: 700;
        color: #6366f1;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
    }
    
    .stat-label {
        font-size: 0.875rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
    
    /* Upload Section */
    .upload-section {
        background: transparent;
        padding: 2.5rem 1.25rem; /* less vertical padding */
        position: relative;
        min-height: unset; /* auto height */
    }
    
    .upload-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.1) 30%, transparent 70%);
        pointer-events: none;
        animation: pulse 6s ease-in-out infinite;
        z-index: -1;
    }
    
    .section-header {
        text-align: center;
        margin-bottom: 1.25rem; /* less gap before uploader */
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.75rem; /* smaller card padding */
        backdrop-filter: blur(20px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .section-header:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateY(-5px);
        box-shadow: 0 35px 70px rgba(0, 0, 0, 0.4);
    }
    
    .section-header h2 {
        font-size: clamp(2rem, 5vw, 3rem);
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #6366f1 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
        text-shadow: 0 0 30px rgba(99, 102, 241, 0.3);
    }
    .section-header p {
        font-size: 1.125rem;
        color: #9ca3af;
        max-width: 500px;
        margin: 0 auto;
        line-height: 1.6;
    }
    /* Prediction Results */
    .prediction-section {
        background: transparent;
        padding: 1.5rem 1.25rem; /* tighter padding */
        position: relative;
        min-height: unset; /* avoid big vertical gap */
        margin-top: 0.5rem;
    }
    .prediction-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.1) 30%, transparent 70%);
        pointer-events: none;
        animation: pulse 6s ease-in-out infinite;
        z-index: -1;
    }
    
    .result-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 3rem;
        max-width: 1200px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
    }
    
    .image-container, .analysis-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 1.25rem;
        backdrop-filter: blur(20px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .image-container:hover, .analysis-container:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateY(-5px);
        box-shadow: 0 35px 70px rgba(0, 0, 0, 0.4);
    }
    
    .image-container h3, .analysis-container h3 {
        font-size: 1.25rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.75rem;
    }

    /* Style the Streamlit image widget to look like our cards */
    [data-testid="stImage"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 0.75rem;
        backdrop-filter: blur(20px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.25);
    }
    [data-testid="stImage"] img {
        width: 100%;
        max-width: 720px; /* keep image within screen */
        height: auto;
        display: block;
        margin-left: auto;
        margin-right: auto; /* center */
        border-radius: 12px;
    }
    
    /* Features Section */
    .features-section {
        background: transparent;
        padding: 6rem 2rem;
        position: relative;
        min-height: 100vh;
    }
    
    .features-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.1) 30%, transparent 70%);
        pointer-events: none;
        animation: pulse 6s ease-in-out infinite;
        z-index: -1;
    }
    
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        max-width: 1200px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(20px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 35px 70px rgba(0, 0, 0, 0.4);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1.5rem;
        opacity: 0.8;
    }
    
    .feature-card h3 {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    
    .feature-card p {
        color: #9ca3af;
        line-height: 1.6;
        font-size: 1rem;
    }
    
    /* Disease Categories */
    .disease-section {
        background: transparent;
        padding: 6rem 2rem;
        position: relative;
        min-height: 100vh;
    }
    
    .disease-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.1) 30%, transparent 70%);
        pointer-events: none;
        animation: pulse 6s ease-in-out infinite;
        z-index: -1;
    }
    
    .disease-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        max-width: 1200px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
    }
    
    .disease-item {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        cursor: pointer;
        color: #ffffff;
        backdrop-filter: blur(20px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
    }
    
    .disease-item:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.3);
        background: rgba(99, 102, 241, 0.05);
        box-shadow: 0 35px 70px rgba(0, 0, 0, 0.4);
    }
    
    /* About Section */
    .about-section {
        background: transparent;
        padding: 6rem 2rem;
        text-align: center;
        position: relative;
        min-height: 100vh;
    }
    
    .about-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.1) 30%, transparent 70%);
        pointer-events: none;
        animation: pulse 6s ease-in-out infinite;
        z-index: -1;
    }
    
    .about-content {
        max-width: 800px;
        margin: 0 auto;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 3rem;
        backdrop-filter: blur(20px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        position: relative;
        z-index: 1;
    }
    
    .about-content:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateY(-5px);
        box-shadow: 0 35px 70px rgba(0, 0, 0, 0.4);
    }
    
    .about-section h2 {
        font-size: clamp(2rem, 5vw, 3rem);
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #6366f1 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        letter-spacing: -0.02em;
        text-shadow: 0 0 30px rgba(99, 102, 241, 0.3);
    }
    
    .about-section p {
        color: #9ca3af;
        line-height: 1.6;
        font-size: 1.125rem;
        margin-bottom: 2rem;
    }
    
    .about-section a {
        color: #6366f1;
        text-decoration: none;
        font-weight: 500;
    }
    
    .about-section a:hover {
        text-decoration: underline;
    }
    
    /* Prediction Card Styles */
    .prediction-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    
    .prediction-card h2 {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    
    .prediction-card h3 {
        font-size: 1.25rem;
        color: #e5e7eb;
        margin-bottom: 1rem;
    }
    
    /* Confidence Badge */
    .confidence-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.875rem;
        margin: 0.5rem 0;
    }
    
    .confidence-high {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .confidence-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    .confidence-low {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    /* Recommendation Boxes */
    .recommendation-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        animation: slideInLeft 0.8s ease-out;
    }
    
    .recommendation-box h4 {
        color: #92400e !important;
        margin-bottom: 1rem;
    }
    
    .recommendation-box ul {
        color: #92400e !important;
        margin: 0;
        padding-left: 1.5rem;
    }
    
    .recommendation-box.info {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-color: #3b82f6;
    }
    
    .recommendation-box.info h4 {
        color: #1e40af !important;
    }
    
    .recommendation-box.info ul {
        color: #1e40af !important;
    }
    
    .recommendation-box.error {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-color: #ef4444;
    }
    
    .recommendation-box.error h4 {
        color: #991b1b !important;
    }
    
    .recommendation-box.error ul {
        color: #991b1b !important;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-section {
            padding: 1rem;
            min-height: 60vh;
        }
        
        .hero-content {
            padding: 1.5rem 1rem;
            margin: 0 0.5rem;
        }
        
        .hero-title {
            font-size: clamp(2rem, 8vw, 3rem);
            white-space: normal;
            line-height: 1.2;
        }
        
        .hero-subtitle {
            font-size: clamp(0.9rem, 4vw, 1.2rem);
            line-height: 1.3;
        }
        
        .hero-description {
            font-size: clamp(0.8rem, 3vw, 0.95rem);
            margin-bottom: 1.5rem;
            padding: 0 0.5rem;
        }
        
        .hero-stats {
            gap: 1rem;
            flex-direction: column;
            align-items: center;
        }
        
        .stat-item {
            padding: 1rem;
            width: 100%;
            max-width: 200px;
        }
        
        .stat-number {
            font-size: 2rem;
        }
        
        .upload-section {
            padding: 1.5rem 0.5rem;
        }
        
        .section-header {
            padding: 1.5rem 1rem;
            margin: 0 0.5rem 1rem 0.5rem;
        }
        
        .section-header h2 {
            font-size: clamp(1.5rem, 6vw, 2.5rem);
        }
        
        .section-header p {
            font-size: 1rem;
            padding: 0 0.5rem;
        }
        
        .stFileUploader {
            margin: 0 0.5rem;
            padding: 1rem;
            max-width: calc(100% - 1rem);
        }
        
        .stFileUploader > div {
            padding: 1.5rem 1rem !important;
        }
        
        .stFileUploader label,
        .stFileUploader p,
        .stFileUploader span {
            font-size: 0.95rem !important;
            line-height: 1.4 !important;
        }
        
        .result-grid {
            grid-template-columns: 1fr;
            gap: 1.5rem;
            padding: 0 0.5rem;
        }
        
        .image-container, .analysis-container {
            padding: 1rem;
        }
        
        .features-grid {
            grid-template-columns: 1fr;
            gap: 1.5rem;
            padding: 0 0.5rem;
        }
        
        .feature-card {
            padding: 1.5rem;
        }
        
        .disease-grid {
            grid-template-columns: 1fr;
            gap: 1rem;
            padding: 0 0.5rem;
        }
        
        .disease-item {
            padding: 1rem;
            font-size: 0.9rem;
        }
        
        .about-content {
            padding: 2rem 1rem;
            margin: 0 0.5rem;
        }
        
        .about-section h2 {
            font-size: clamp(1.5rem, 6vw, 2.5rem);
        }
        
        .about-section p {
            font-size: 1rem;
            padding: 0 0.5rem;
        }
    }
    
    @media (max-width: 480px) {
        .hero-title {
            font-size: clamp(1.8rem, 10vw, 2.5rem);
        }
        
        .hero-subtitle {
            font-size: clamp(0.8rem, 5vw, 1rem);
        }
        
        .hero-description {
            font-size: clamp(0.75rem, 4vw, 0.9rem);
        }
        
        .section-header h2 {
            font-size: clamp(1.2rem, 8vw, 2rem);
        }
        
        .stFileUploader {
            margin: 0 0.25rem;
        }
        
        .stFileUploader > div {
            padding: 1rem 0.5rem !important;
        }
        
        .stFileUploader label,
        .stFileUploader p,
        .stFileUploader span {
            font-size: 0.9rem !important;
        }
        
        .stat-item {
            padding: 0.75rem;
        }
        
        .stat-number {
            font-size: 1.5rem;
        }
        
        .stat-label {
            font-size: 0.75rem;
        }
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #6366f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #8b5cf6;
    }
    
    /* Animations */
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Ripple animation */
    @keyframes ripple {
        0% {
            transform: scale(0);
            opacity: 1;
        }
        100% {
            transform: scale(4);
            opacity: 0;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        detector = SkinDiseaseDetector()
        detector.get_class_names('class_names.txt')
        detector.load_model(MODEL_PATH)
        return detector
    except Exception as e:
        st.error(f"Model not found or failed to load. Error: {e}")
        return None

def preprocess_image(image):
    """Preprocess uploaded image"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_confidence_badge(confidence):
    """Get confidence badge HTML"""
    if confidence >= 0.8:
        return f'<span class="confidence-badge confidence-high">High Confidence: {confidence:.1%}</span>'
    elif confidence >= 0.6:
        return f'<span class="confidence-badge confidence-medium">Medium Confidence: {confidence:.1%}</span>'
    else:
        return f'<span class="confidence-badge confidence-low">Low Confidence: {confidence:.1%}</span>'

def get_confidence_color(confidence):
    """Get confidence color class"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

def main():
    # FORMLESS-Style Background
    st.markdown("""
    <div class="formless-background"></div>
    <div class="background-overlay"></div>
    <div class="cursor-reactive-bg"></div>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-content">
            <h1 class="hero-title">DermoraSense</h1>
            <p class="hero-subtitle">Next-Gen Deep Learning–Powered Skin Disease Detection</p>
            <p class="hero-description">
                Experience the future of dermatological analysis with our cutting-edge deep learning technology. 
                Get instant, accurate diagnoses powered by state-of-the-art neural networks.
            </p>
            <div class="hero-stats">
                <div class="stat-item">
                    <span class="stat-number">95.89%</span>
                    <span class="stat-label">Accuracy</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">23</span>
                    <span class="stat-label">Diseases</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">24/7</span>
                    <span class="stat-label">Available</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Upload Section with integrated file uploader
    st.markdown("""
    <div class="upload-section">
        <div class="section-header">
            <h2>Get Started in Seconds</h2>
            <p>Upload a clear image of the skin condition and receive instant Analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Working file uploader integrated in the upload section
    st.markdown("""
    <div style="text-align: center; margin: 0.75rem 0 0.75rem; padding: 0 1rem;">
        <p style="color: #cbd5e1; margin-bottom: 0.5rem; font-size: 0.9rem;">Upload your image below</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drag and drop your image here, or click to browse",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of the skin condition for best results",
        label_visibility="visible"
    )
    
    # Prediction Section
    if uploaded_file is not None:
        # Load model
        detector = load_model()
        if detector is None:
            st.error("❌ Model failed to load. Please refresh the page.")
            st.stop()
        
        # Display uploaded image and analysis
        st.markdown("""
        <div class="prediction-section">
            <div class="result-grid">
                <div class="image-container">
                    <h3>📷 Uploaded Image</h3>
        """, unsafe_allow_html=True)
        
        image = Image.open(uploaded_file)
        image_for_pred = image.copy()
        st.image(image, caption="Uploaded Image", use_container_width=False, width=720)
        
        st.markdown("""
                </div>
                <div class="analysis-container">
                    <h3>🔍 AI Analysis</h3>
        """, unsafe_allow_html=True)
        
        # Preprocess and predict
        with st.spinner("🔎 Analyzing your image..."):
            img_array = preprocess_image(image_for_pred)
            prediction = detector.model.predict(img_array)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            disease_name = detector.class_names[predicted_class]
        
        # Display prediction results
        st.markdown(f"""
                    <div class="prediction-card">
                        <h2>🎯 Diagnosis Result</h2>
                        <h3>{disease_name}</h3>
                        {get_confidence_badge(confidence)}
                    </div>
        """, unsafe_allow_html=True)
        
        # Top 5 predictions chart
        st.markdown("""
                    <h3 style="margin-top: 2rem; margin-bottom: 1rem;">📊 Prediction Breakdown</h3>
        """, unsafe_allow_html=True)
        
        top_5_indices = np.argsort(prediction[0])[-5:][::-1]
        
        fig = go.Figure(data=[
            go.Bar(
                x=[detector.class_names[i] for i in top_5_indices],
                y=[prediction[0][i] for i in top_5_indices],
                marker_color=['#6366f1' if i == predicted_class else '#6b7280' for i in top_5_indices],
                text=[f'{prediction[0][i]:.1%}' for i in top_5_indices],
                textposition='auto',
            )
        ])
        fig.update_layout(
            title="",
            xaxis_title="",
            yaxis_title="",
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, color='#9ca3af'),
            yaxis=dict(showgrid=False, zeroline=False, color='#9ca3af'),
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        if confidence >= 0.8:
            st.markdown("""
                    <div class="recommendation-box info">
                        <h4 style="color: #1e40af; margin-bottom: 1rem;">✅ High Confidence Prediction</h4>
                        <ul style="color: #1e40af; margin: 0; padding-left: 1.5rem;">
                            <li>Consider consulting a dermatologist for confirmation</li>
                            <li>Monitor the condition for any changes</li>
                            <li>Follow appropriate treatment guidelines</li>
                        </ul>
                    </div>
            """, unsafe_allow_html=True)
        elif confidence >= 0.6:
            st.markdown("""
                    <div class="recommendation-box">
                        <h4 style="color: #92400e; margin-bottom: 1rem;">⚠️ Medium Confidence Prediction</h4>
                        <ul style="color: #92400e; margin: 0; padding-left: 1.5rem;">
                            <li>Multiple conditions may have similar symptoms</li>
                            <li>Professional medical evaluation is recommended</li>
                            <li>Consider additional diagnostic tests</li>
                        </ul>
                    </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                    <div class="recommendation-box error">
                        <h4 style="color: #991b1b; margin-bottom: 1rem;">❌ Low Confidence Prediction</h4>
                        <ul style="color: #991b1b; margin: 0; padding-left: 1.5rem;">
                            <li>Image quality or lighting may be affecting results</li>
                            <li>Try uploading a clearer, better-lit image</li>
                            <li>Professional medical evaluation is strongly recommended</li>
                        </ul>
                    </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown("""
    <div class="features-section">
        <div class="section-header">
            <h2>Why Choose DermoraSense?</h2>
            <p>Built with cutting-edge technology and designed for medical professionals</p>
        </div>
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <h3>High Accuracy</h3>
                <p>Powered by ResNet50V2 transfer learning with 95.9% accuracy across 23 disease categories.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <h3>Instant Results</h3>
                <p>Get diagnosis results in seconds with our optimized deep learning model.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🔒</div>
                <h3>Privacy First</h3>
                <p>Your images are processed locally and never stored or shared.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📱</div>
                <h3>Device Agnostic</h3>
                <p>Works seamlessly on any device - desktop, tablet, or mobile.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Supported Diseases
    st.markdown("""
    <div class="disease-section">
        <div class="section-header">
            <h2>Supported Disease Categories</h2>
            <p>Our Deep Learning model can detect and classify 23 different skin conditions with high accuracy</p>
        </div>
        <div class="disease-grid">
            <div class="disease-item">🔴 Acne and Rosacea</div>
            <div class="disease-item">🟠 Atopic Dermatitis</div>
            <div class="disease-item">🟡 Bullous Disease</div>
            <div class="disease-item">🟢 Cellulitis & Bacterial Infections</div>
            <div class="disease-item">🔵 Eczema</div>
            <div class="disease-item">🟣 Exanthems & Drug Eruptions</div>
            <div class="disease-item">⚫ Hair Loss & Alopecia</div>
            <div class="disease-item">⚪ Herpes & STDs</div>
            <div class="disease-item">🟤 Light Diseases & Pigmentation</div>
            <div class="disease-item">🔴 Lupus & Connective Tissue</div>
            <div class="disease-item">🟠 Melanoma & Skin Cancer</div>
            <div class="disease-item">🟡 Nail Fungus & Nail Diseases</div>
            <div class="disease-item">🟢 Poison Ivy & Contact Dermatitis</div>
            <div class="disease-item">🔵 Psoriasis & Lichen Planus</div>
            <div class="disease-item">🟣 Scabies & Infestations</div>
            <div class="disease-item">⚫ Seborrheic Keratoses</div>
            <div class="disease-item">⚪ Systemic Disease</div>
            <div class="disease-item">🟤 Tinea & Fungal Infections</div>
            <div class="disease-item">🔴 Urticaria & Hives</div>
            <div class="disease-item">🟠 Vascular Tumors</div>
            <div class="disease-item">🟡 Vasculitis</div>
            <div class="disease-item">🟢 Viral Infections</div>
            <div class="disease-item">🔵 Actinic Keratosis & Malignant Lesions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # About Section
    st.markdown("""
    <div class="about-section">
        <div class="about-content">
            <h2>About DermoraSense</h2>
            <p>
                DermoraSense, a cutting-edge dermatological AI technology, was developed by a team of AI researchers to assist in the preliminary identification of common skin conditions. Our platform seamlessly integrates deep learning expertise with medical domain knowledge, ensuring accurate and reliable skin disease detection.
            </p>
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon">🔬</div>
                    <h3>Technology</h3>
                    <p>Built with TensorFlow 2.x, ResNet50V2 architecture, and advanced image processing techniques.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <h3>Dataset</h3>
                    <p>Trained on a comprehensive dataset of 23 skin disease categories with thousands of high-quality images.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">⚠️</div>
                    <h3>Disclaimer</h3>
                    <p>This tool is for educational purposes only and should not replace professional medical advice.</p>
                </div>
            </div>
            <div style="text-align: center; margin-top: 3rem;">
                <p style="margin-bottom: 1rem;">
                    <strong>Contact:</strong> maheshvellogi99@gmail.com
                </p>
                <p>
                    <strong>GitHub:</strong> <a href="https://github.com/maheshvellogi99">maheshvellogi99</a> | 
                    <strong>LinkedIn:</strong> <a href="https://www.linkedin.com/in/mahesh-vellogi-aa13522ba/">Mahesh Vellogi</a>
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    
    # Add JavaScript for cursor-reactive background using Streamlit's proper method
    html("""
    <script>
        // FORMLESS-Style Particle System
        class ParticleSystem {
            constructor() {
                this.particles = [];
                this.mouseX = 0;
                this.mouseY = 0;
                this.init();
            }
            
            init() {
                this.createParticles();
                this.bindEvents();
                this.animate();
            }
            
            createParticles() {
                const particleCount = 80;
                const container = document.querySelector('.formless-background');
                if (!container) return;
                
                for (let i = 0; i < particleCount; i++) {
                    const particle = document.createElement('div');
                    particle.className = 'particle';
                    
                    // Random size between 3px and 12px
                    const size = Math.random() * 9 + 3;
                    particle.style.width = size + 'px';
                    particle.style.height = size + 'px';
                    
                    // Random position
                    particle.style.left = Math.random() * 100 + '%';
                    particle.style.top = Math.random() * 100 + '%';
                    
                    // Random animation delay
                    particle.style.animationDelay = Math.random() * 8 + 's';
                    
                    container.appendChild(particle);
                    this.particles.push(particle);
                }
            }
            
            bindEvents() {
                document.addEventListener('mousemove', (e) => {
                    this.mouseX = e.clientX;
                    this.mouseY = e.clientY;
                    this.updateParticles();
                    this.updateCursorReactiveBg();
                });
                
                document.addEventListener('touchmove', (e) => {
                    if (e.touches[0]) {
                        this.mouseX = e.touches[0].clientX;
                        this.mouseY = e.touches[0].clientY;
                        this.updateParticles();
                        this.updateCursorReactiveBg();
                    }
                });
                
                // Add cursor trail effect
                this.createCursorTrail();
                
                // Add click ripple effect
                document.addEventListener('click', (e) => {
                    this.createRipple(e.clientX, e.clientY);
                });
            }
            
            updateCursorReactiveBg() {
                const cursorBg = document.querySelector('.cursor-reactive-bg');
                if (cursorBg) {
                    cursorBg.style.setProperty('--mouse-x', this.mouseX + 'px');
                    cursorBg.style.setProperty('--mouse-y', this.mouseY + 'px');
                }
            }
            
            createCursorTrail() {
                const trail = document.createElement('div');
                trail.className = 'cursor-trail';
                document.body.appendChild(trail);
                
                document.addEventListener('mousemove', (e) => {
                    trail.style.left = e.clientX - 3 + 'px';
                    trail.style.top = e.clientY - 3 + 'px';
                });
            }
            
            createRipple(x, y) {
                const ripple = document.createElement('div');
                ripple.style.position = 'fixed';
                ripple.style.left = (x - 25) + 'px';
                ripple.style.top = (y - 25) + 'px';
                ripple.style.width = '50px';
                ripple.style.height = '50px';
                ripple.style.border = '2px solid rgba(99, 102, 241, 0.6)';
                ripple.style.borderRadius = '50%';
                ripple.style.pointerEvents = 'none';
                ripple.style.zIndex = '9998';
                ripple.style.animation = 'ripple 0.8s ease-out forwards';
                
                document.body.appendChild(ripple);
                
                setTimeout(() => {
                    document.body.removeChild(ripple);
                }, 800);
            }
            
            updateParticles() {
                this.particles.forEach((particle, index) => {
                    const rect = particle.getBoundingClientRect();
                    const particleX = rect.left + rect.width / 2;
                    const particleY = rect.top + rect.height / 2;
                    
                    // Calculate distance from mouse
                    const deltaX = this.mouseX - particleX;
                    const deltaY = this.mouseY - particleY;
                    const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
                    
                    // React to mouse proximity with magnetic effect
                    if (distance < 200) {
                        const force = (200 - distance) / 200;
                        const moveX = deltaX * force * 0.02;
                        const moveY = deltaY * force * 0.02;
                        
                        // Apply magnetic movement
                        const currentLeft = parseFloat(particle.style.left);
                        const currentTop = parseFloat(particle.style.top);
                        
                        particle.style.left = (currentLeft + moveX) + '%';
                        particle.style.top = (currentTop + moveY) + '%';
                        
                        // Enhanced visual effects
                        particle.style.opacity = 0.4 + force * 0.6;
                        particle.style.transform = `scale(${1 + force * 0.5})`;
                        particle.classList.add('magnetic', 'active');
                        
                        // Add glow effect
                        particle.style.boxShadow = `0 0 ${30 + force * 40}px rgba(99, 102, 241, ${0.4 + force * 0.6})`;
                    } else {
                        // Gradually return to normal
                        particle.style.opacity = 0.4;
                        particle.style.transform = 'scale(1)';
                        particle.classList.remove('magnetic', 'active');
                        particle.style.boxShadow = 'none';
                    }
                });
            }
            
            animate() {
                // Continuous subtle movement
                this.particles.forEach((particle, index) => {
                    const time = Date.now() * 0.001;
                    const offset = index * 0.1;
                    
                    // Add subtle wave motion
                    const waveX = Math.sin(time + offset) * 0.8;
                    const waveY = Math.cos(time + offset * 0.7) * 0.8;
                    
                    const currentLeft = parseFloat(particle.style.left);
                    const currentTop = parseFloat(particle.style.top);
                    
                    particle.style.left = (currentLeft + waveX) + '%';
                    particle.style.top = (currentTop + waveY) + '%';
                });
                
                requestAnimationFrame(() => this.animate());
            }
        }
        
        // Initialize particle system when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new ParticleSystem();
        });
        
        // Fallback for Streamlit
        if (typeof window !== 'undefined') {
            setTimeout(() => {
                if (!document.querySelector('.particle')) {
                    new ParticleSystem();
                }
            }, 1000);
        }
    </script>
    """, height=0)
