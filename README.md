# Skin Disease Detection System

A comprehensive Deep learning system for detecting and classifying 23 different types of skin diseases using deep learning.

# Try the Live App Here:
https://dermorasense.streamlit.app/

## Features

- **23 Disease Categories**: Comprehensive coverage of common skin conditions
- **High Accuracy**: Powered by ResNet50V2 transfer learning
- **Easy-to-Use Interface**: Streamlit web application with drag-and-drop functionality
- **Detailed Analysis**: Confidence scores and probability distributions
- **Professional UI**: Modern, responsive design with medical-grade interface

## Supported Diseases

1. **Acne and Rosacea Photos**
2. **Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions**
3. **Atopic Dermatitis Photos**
4. **Bullous Disease Photos**
5. **Cellulitis Impetigo and other Bacterial Infections**
6. **Eczema Photos**
7. **Exanthems and Drug Eruptions**
8. **Hair Loss Photos Alopecia and other Hair Diseases**
9. **Herpes HPV and other STDs Photos**
10. **Light Diseases and Disorders of Pigmentation**
11. **Lupus and other Connective Tissue diseases**
12. **Melanoma Skin Cancer Nevi and Moles**
13. **Nail Fungus and other Nail Disease**
14. **Poison Ivy Photos and other Contact Dermatitis**
15. **Psoriasis pictures Lichen Planus and related diseases**
16. **Scabies Lyme Disease and other Infestations and Bites**
17. **Seborrheic Keratoses and other Benign Tumors**
18. **Systemic Disease**
19. **Tinea Ringworm Candidiasis and other Fungal Infections**
20. **Urticaria Hives**
21. **Vascular Tumors**
22. **Vasculitis Photos**
23. **Warts Molluscum and other Viral Infections**

## Quick Start

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.x
- CUDA-compatible GPU (recommended for faster training)
  
## Usage

1. Open the app in your browser - 
2. Navigate to the **Upload & Predict** page.
3. Upload a clear image of the skin condition.
4. View the predicted disease, confidence score, and recommendations.

## Model Architecture

### Base Model
- **ResNet50V2**: Pre-trained on ImageNet
- **Transfer Learning**: Leverages pre-trained weights for better performance

### Custom Layers
```
ResNet50V2 (base)
├── Global Average Pooling
├── Dropout (0.5)
├── Dense (512, ReLU)
├── Dropout (0.3)
├── Dense (256, ReLU)
├── Dropout (0.2)
└── Dense (23, Softmax)
```

### Training Strategy
- **Two-phase training**: Feature extraction + Fine-tuning
- **Data augmentation**: Rotation, shift, shear, zoom, flip
- **Regularization**: Dropout layers to prevent overfitting
- **Callbacks**: Early stopping, learning rate reduction, model checkpointing

## Performance

The model achieves high accuracy across multiple skin disease categories with:
- **Transfer Learning**: Leverages ResNet50V2 pre-trained weights
- **Data Augmentation**: Improves generalization
- **Regularization**: Prevents overfitting
- **Fine-tuning**: Optimizes for specific disease detection

## Configuration

### Model Parameters
- **Image Size**: 224x224 pixels
- **Batch Size**: 32 (configurable)
- **Learning Rate**: 0.001 (initial), 0.0001 (fine-tuning)
- **Epochs**: 30 (initial training) + 20 (fine-tuning)

### Data Augmentation
- **Rotation**: ±20 degrees
- **Width/Height Shift**: ±20%
- **Shear**: ±20%
- **Zoom**: ±20%
- **Horizontal Flip**: Enabled

## Important Notes

### Medical Disclaimer
- **This tool is for educational purposes only**
- **Not a replacement for professional medical advice**
- **Always consult healthcare professionals for diagnosis**
- **Results should be validated by medical experts**

### Limitations
- Accuracy depends on image quality and lighting
- Some rare conditions may not be accurately classified
- Model performance varies across different skin types
- Results are probabilistic and not definitive

### Best Practices
- Use clear, well-lit images
- Ensure good image resolution
- Include the affected area clearly
- Avoid shadows and reflections
- Use consistent lighting conditions

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**:
   ```bash
   # Install CPU-only TensorFlow
   pip install tensorflow-cpu
   ```

2. **Memory Issues**:
   - Reduce batch size in `skin_disease_model.py`
   - Use smaller image size
   - Close other applications

3. **Model Not Found**:
   - Ensure you've trained the model first
   - Check file path in `app.py`

4. **Dataset Issues**:
   - Verify folder structure matches expected format
   - Ensure all categories have images
   - Check file permissions

## Future Enhancements
   - Mobile application development
   - Real-time video analysis
   - Integration with electronic health records
   - Additional disease categories
   - Multi-language support
   - API development for third-party integration
## Support

For questions, issues, or contributions:
- Contact me at Github: https://github.com/maheshvellogi99 or at Linkedin: http://linkedin.com/in/mahesh-vellogi-aa13522ba
- Check the troubleshooting section
- Review the code comments
- Ensure all dependencies are installed
- Verify dataset structure

##� License

This project is for educational and research purposes. Please ensure compliance with local regulations regarding medical software and data privacy.

---

**Note**: This system is designed to assist healthcare professionals and should not be used as a standalone diagnostic tool. Always consult qualified medical professionals for proper diagnosis and treatment. 

## Outputs:
<img width="1236" height="1041" alt="Screenshot 2025-09-14 at 9 36 17 PM" src="https://github.com/user-attachments/assets/096da653-9cd8-4128-b474-22628036a1f4" />
<img width="1236" height="1041" alt="Screenshot 2025-09-14 at 9 38 04 PM" src="https://github.com/user-attachments/assets/a514880b-c1c8-47ab-82f2-3ea321bc295f" />

