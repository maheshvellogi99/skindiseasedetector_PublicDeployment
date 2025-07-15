# 🏥 Skin Disease Detection System

An AI-powered web application that detects and classifies 23 different skin diseases using deep learning.

## 🌟 Features

- **23 Disease Categories**: Comprehensive coverage of common skin conditions
- **High Accuracy**: Powered by custom model designed using ResNet50V2 and VGG16 transfer learning
- **Mobile Responsive**: Works on all devices and browsers
- **Real-time Analysis**: Instant predictions with confidence scores
- **Detailed Reports**: Top 5 predictions with probability distributions

## 🚀 Live Demo

**Deployed on Streamlit Cloud**: [Your App URL will be here after deployment]

## 📱 Mobile Access

The application is fully responsive and can be accessed from:
- Mobile browsers (Chrome, Safari, Firefox)
- Tablet browsers
- Desktop browsers

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow 2.x with Keras
- **Model**: ResNet50V2 with transfer learning
- **Web Framework**: Streamlit
- **Image Processing**: OpenCV, PIL
- **Data Visualization**: Plotly, Matplotlib

## 📊 Supported Diseases

1. Acne and Rosacea
2. Actinic Keratosis and Malignant Lesions
3. Atopic Dermatitis
4. Bullous Disease
5. Cellulitis and Bacterial Infections
6. Eczema
7. Exanthems and Drug Eruptions
8. Hair Loss and Alopecia
9. Herpes and STDs
10. Light Diseases and Pigmentation Disorders
11. Lupus and Connective Tissue Diseases
12. Melanoma and Skin Cancer
13. Nail Fungus and Nail Diseases
14.Viral Infections (Warts, Molluscum)
15. Poison Ivy and Contact Dermatitis
16. Psoriasis and Lichen Planus
17. Scabies and Infestations
18. Seborrheic Keratoses and Benign Tumors
19. Systemic Disease
20. Tinea and Fungal Infections
21. Urticaria and Hives
22. Vascular Tumors
23. Vasculitis

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/skin-disease-detection.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set the path to your app: `app.py`
   - Click "Deploy"

### Option 2: Heroku

1. **Create Procfile**:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Deploy**:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 3: Railway

1. **Connect to Railway**:
   - Go to [railway.app](https://railway.app)
   - Connect your GitHub repository
   - Railway will automatically detect and deploy your Streamlit app

### Option 4: Google Cloud Run

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Deploy**:
   ```bash
   gcloud run deploy skin-disease-detection --source .
   ```

## 📱 Mobile Optimization

The app is already optimized for mobile devices with:
- Responsive design
- Touch-friendly interface
- Optimized image upload
- Fast loading times

## 🔒 Privacy & Security

- Images are processed locally and not stored
- No personal health information is collected
- All predictions are temporary and not saved
- HTTPS encryption for all data transmission

## ⚠️ Medical Disclaimer

This tool is for educational purposes only and should not replace professional medical advice. Always consult with a healthcare professional for proper diagnosis and treatment.

## 📞 Contact

- **Email**: maheshvellogi99@gmail.com
- **GitHub**: https://github.com/maheshvellogi99
- **LinkedIn**: https://www.linkedin.com/in/mahesh-vellogi-aa13522ba/

## 📈 Performance

- **Model Accuracy**: High accuracy across multiple skin disease categories
- **Response Time**: < 5 seconds for image analysis
- **Supported Formats**: JPG, PNG, JPEG
- **Image Size**: Optimized for 224x224 pixels

## 🔄 Updates

- Regular model updates for improved accuracy
- New disease categories added periodically
- Performance optimizations for faster processing 