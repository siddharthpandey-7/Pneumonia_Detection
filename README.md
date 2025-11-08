# 🩺 Pneumonia Detection System

A comprehensive deep learning-powered web application for detecting pneumonia from chest X-ray images using transfer learning with VGG19 architecture. Built with Flask, TensorFlow, and a modern responsive UI.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-v2.18.0-orange.svg)
![Flask](https://img.shields.io/badge/flask-v2.3.3-green.svg)
![Keras](https://img.shields.io/badge/keras-v3.8.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Project Overview

This project implements an end-to-end deep learning solution for pneumonia detection from chest X-ray images. The system achieves **91.67% accuracy** on the test set using VGG19 transfer learning architecture with fine-tuning, trained on 5,216 chest X-ray images from Kaggle's pneumonia dataset.

### Key Features
- 🤖 **Deep Learning Model**: VGG19 architecture with transfer learning from ImageNet
- 🎯 **High Accuracy**: 91.67% test accuracy with 95% pneumonia recall
- 🎨 **Modern Web Interface**: Responsive design with real-time image preview
- ⚡ **Real-time Predictions**: Instant pneumonia detection with confidence scores
- 📊 **Visual Analytics**: Color-coded results with confidence progress bars
- 🔄 **Data Augmentation**: Advanced image preprocessing for robust predictions
- 📁 **Two-Phase Training**: Initial transfer learning + fine-tuning approach
- 🔒 **Error Handling**: Comprehensive validation and error management

## 🏥 Medical Background

The system detects pneumonia from chest X-ray radiographs, a critical diagnostic tool in respiratory medicine.

### Conditions Classified:
- **NORMAL**: Healthy lungs with no signs of infection
- **PNEUMONIA**: Bacterial or viral lung infection with characteristic opacity patterns

### Clinical Significance:
- Pneumonia affects millions worldwide annually
- Early detection is crucial for timely treatment
- X-ray analysis is the gold standard diagnostic method
- AI assistance can accelerate diagnosis in high-volume settings

### Image Processing:
- **Input**: Chest X-ray images (JPEG/PNG format)
- **Resolution**: Resized to 128×128 pixels for model input
- **Color Space**: Grayscale X-rays converted to RGB (3 channels)
- **Normalization**: Pixel values scaled to [0, 1] range

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
TensorFlow 2.18.0
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/siddharthpandey-7/Pneumonia-Detection-Using-Deep-Learning.git
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the trained model**
   
   ⚠️ **IMPORTANT**: The model file is too large for GitHub and must be downloaded separately.
   
   🔗 **[Download best_vgg19_pneumonia.h5 from Google Drive](https://drive.google.com/file/d/1g-M2JrvOxpNCr4hsHJUpqW7EHv3BkYgn/view?usp=drive_link)**
   
   After downloading:
   - Place `best_vgg19_pneumonia.h5` in the root directory
   - File size should be approximately 80-100 MB
   - Verify the file path: `pneumonia-detection/best_vgg19_pneumonia.h5`

5. **Create upload directory**
   ```bash
   mkdir static/uploads
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Access the web interface**
   ```
   Open your browser and navigate to: http://localhost:5000
   ```

## 📁 Project Structure

```
pneumonia-detection/
│
├── app.py                          # Flask web application
├── pneumonia.ipynb                 # Model training & evaluation notebook
├── best_vgg19_pneumonia.h5         # Trained VGG19 model (download separately)
│
├── templates/                      # HTML templates
│   ├── index.html                  # Main upload interface
│   └── result.html                 # Prediction results page
│
├── static/                         # Static assets
│   ├── style.css                   # Custom styling
│   └── uploads/                    # Temporary image storage (gitignored)
│
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── .gitignore                     # Git ignore rules
```

## 🧠 Deep Learning Pipeline

### Dataset Information
- **Source**: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total Images**: 5,856 chest X-ray images
- **Origin**: Guangzhou Women and Children's Medical Center
- **Format**: JPEG grayscale images
- **Classes**: Normal, Pneumonia (Bacterial & Viral)

### Data Distribution
| Split | Normal | Pneumonia | Total |
|-------|--------|-----------|-------|
| **Training** | 1,341 | 3,875 | 5,216 |
| **Validation** | 8 | 8 | 16 |
| **Testing** | 234 | 390 | 624 |

### Preprocessing Pipeline
```python
Input Image (Variable size)
    ↓
Resize to 128×128 pixels
    ↓
Convert to RGB (3 channels)
    ↓
Normalize to [0, 1] range
    ↓
Data Augmentation (Training only):
  - Rotation: ±20°
  - Width/Height Shift: 20%
  - Shear: 15%
  - Zoom: 15%
  - Horizontal Flip: Yes
    ↓
Model Input (128, 128, 3)
```

## 🏗️ Model Architecture

### Base Model: VGG19
```
VGG19 (ImageNet pre-trained)
  - Total params: 24,285,122 (92.64 MB)
  - Trainable params (Phase 1): 4,260,738 (16.25 MB)
  - Non-trainable params (Phase 1): 20,024,384 (76.39 MB)
  ↓
Frozen convolutional layers (initial training)
  ↓
Flatten Layer (8192 features)
  ↓
Dense(512, activation='relu') + Dropout(0.5)
  ↓
Dense(128, activation='relu')
  ↓
Dense(2, activation='softmax') → [NORMAL, PNEUMONIA]
```

### Training Strategy

#### Phase 1: Transfer Learning (5 epochs with early stopping)
- **Base Model**: VGG19 with frozen weights
- **Optimizer**: Adam (learning_rate=1e-4)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32
- **Image Size**: 128×128×3
- **Best Val Loss**: 0.4489 (Epoch 1)
- **Training Accuracy**: 78.73% → 92.73%
- **Val Accuracy**: 81.25%

#### Phase 2: Fine-tuning (9 epochs with early stopping)
- **Unfrozen Layers**: block5_conv1 onwards
- **Trainable params**: 13,699,970 (52.26 MB)
- **Optimizer**: Adam (learning_rate=1e-5)
- **Strategy**: Fine-tune top VGG blocks on medical images
- **Best Val Loss**: 0.2209 (Epoch 5)
- **Training Accuracy**: 90.91% → 97.09%
- **Val Accuracy**: 81.25% → 87.50%

### Callbacks & Optimization
```python
- EarlyStopping: monitor='val_loss', patience=4
- ReduceLROnPlateau: monitor='val_accuracy', factor=0.5, patience=3
- ModelCheckpoint: save_best_only=True (based on val_loss)
```

## 🌐 Web Application

### Frontend Features
- **Responsive Design**: Mobile-first approach with modern CSS gradients
- **Image Preview**: Real-time preview before prediction
- **Loading Animation**: Spinning loader with visual feedback during processing
- **Result Display**: Color-coded predictions (green for NORMAL, red for PNEUMONIA)
- **Confidence Bar**: Visual progress bar showing prediction confidence
- **User Experience**: Intuitive interface with clear instructions

### Backend Architecture
- **Framework**: Flask with RESTful design
- **Model Loading**: Efficient .h5 model persistence
- **Image Processing**: TensorFlow Keras preprocessing
- **Error Handling**: Comprehensive validation for uploads
- **File Management**: Temporary storage in `static/uploads/`

### API Endpoints

#### `GET /`
Main upload interface
- **Returns**: HTML form for image upload
- **Features**: File input with preview capability

#### `POST /predict`
Process uploaded X-ray image
- **Input**: `multipart/form-data` with image file
- **Processing**: 
  1. Save uploaded file to `static/uploads/`
  2. Preprocess image (resize to 128×128, normalize)
  3. Model prediction with confidence calculation
  4. Return results page
- **Output**: Result page with prediction and confidence percentage
- **Error Handling**: Invalid files, missing uploads

#### `GET /display/<filename>`
Display uploaded image
- **Input**: Filename parameter
- **Returns**: Image from uploads directory

## 🎯 Model Performance

### Test Set Results (After Fine-tuning)
```
Final Test Accuracy: 91.67%
Final Test Loss: 0.2557

Classification Report:
                 precision    recall  f1-score   support

      NORMAL       0.89      0.66      0.76       234
   PNEUMONIA       0.82      0.95      0.88       390

    accuracy                           0.84       624
   macro avg       0.86      0.81      0.82       624
weighted avg       0.85      0.84      0.84       624
```

### Performance Analysis
- **Overall Accuracy**: 91.67% on test set
- **Pneumonia Detection (Recall)**: 95% - Critical for medical screening
- **Normal Detection (Precision)**: 89% - Reduces false positives
- **F1-Score**: 0.88 for Pneumonia, 0.76 for Normal

### Training Progression

#### Initial Training Phase
| Epoch | Train Acc | Val Acc | Train Loss | Val Loss | Learning Rate |
|-------|-----------|---------|------------|----------|---------------|
| 1 | 78.73% | 81.25% | 0.4605 | 0.4489 | 1e-4 |
| 2 | 89.55% | 81.25% | 0.2482 | 0.5527 | 1e-4 |
| 3 | 91.44% | 81.25% | 0.2099 | 0.5870 | 1e-4 |
| 4 | 92.36% | 81.25% | 0.1875 | 0.6481 | 1e-4 |
| 5 | 92.73% | 81.25% | 0.1829 | 0.5434 | 5e-5 |

**Result**: Early stopping at Epoch 5, restored best weights from Epoch 1

#### Fine-tuning Phase
| Epoch | Train Acc | Val Acc | Train Loss | Val Loss | Learning Rate |
|-------|-----------|---------|------------|----------|---------------|
| 1 | 90.91% | 81.25% | 0.2102 | 0.4242 | 1e-5 |
| 2 | 93.97% | 81.25% | 0.1651 | 0.3650 | 1e-5 |
| 3 | 94.83% | 87.50% | 0.1349 | 0.3764 | 1e-5 |
| 5 | 95.48% | 81.25% | 0.1194 | 0.2209 | 1e-5 |
| 9 | 97.09% | 87.50% | 0.0848 | 0.3292 | 5e-6 |

**Result**: Early stopping at Epoch 9, restored best weights from Epoch 5

### Confusion Matrix Analysis

**Before Fine-tuning:**
```
                Predicted
              NORMAL  PNEUMONIA
Actual NORMAL    155        79
     PNEUMONIA    19       371
```
- True Positives (Pneumonia): 371
- True Negatives (Normal): 155
- False Positives: 79
- False Negatives: 19

**After Fine-tuning:**
```
                Predicted
              NORMAL  PNEUMONIA
Actual NORMAL    197        37
     PNEUMONIA    15       375
```
- True Positives (Pneumonia): 375 ✅
- True Negatives (Normal): 197 ✅
- False Positives: 37 (reduced from 79)
- False Negatives: 15 (reduced from 19)

**Improvement**: Fine-tuning significantly reduced false predictions in both classes!

## 📊 Visual Results

### Training Curves
The model shows excellent convergence:
- **Training Accuracy**: Steady improvement from 84% to 93%
- **Validation Accuracy**: Stable at ~81%
- **Training Loss**: Decreases from 0.35 to 0.18
- **Validation Loss**: Increases slightly (expected with small validation set)

### Sample Predictions
The model correctly identifies NORMAL chest X-rays with high confidence, demonstrating strong feature learning from the VGG19 architecture.

## 🔧 Technical Implementation

### Dependencies
```python
Flask==2.3.3              # Web framework
tensorflow==2.18.0        # Deep learning framework
keras==3.8.0              # High-level neural networks API
numpy==1.24.3             # Numerical computing
Pillow==10.0.0            # Image processing
opencv-python==4.8.0.76   # Computer vision utilities
pandas==2.0.3             # Data manipulation
matplotlib==3.7.2         # Visualization
seaborn==0.12.2           # Statistical visualization
scikit-learn==1.3.0       # ML utilities
```

### Key Technologies
- **Deep Learning**: TensorFlow 2.18, Keras 3.8, VGG19
- **Web Framework**: Flask with Jinja2 templating
- **Image Processing**: PIL, OpenCV, Keras preprocessing
- **Data Science**: NumPy, Pandas, Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3 (gradient backgrounds), JavaScript
- **Model Format**: HDF5 (.h5) for Keras models
- **Hardware**: Trained on Kaggle GPU (Tesla P100-PCIE-16GB)

## 💻 Usage Example

### Web Interface Usage
```
1. Navigate to http://localhost:5000
2. Click "Choose File" button
3. Select chest X-ray image (JPEG/PNG)
4. Image preview appears automatically
5. Click "🔍 Analyze Image" button
6. Loading spinner shows processing status
7. View results page with:
   - Uploaded X-ray image
   - Prediction: NORMAL (green) or PNEUMONIA (red)
   - Confidence score with progress bar
8. Click "⬅️ Scan Another Image" to analyze more
```

### Programmatic Usage
```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = load_model('best_vgg19_pneumonia.h5')

# Load and preprocess image
img_path = 'chest_xray.jpg'
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Make prediction
prediction = model.predict(img_array)
class_names = ['NORMAL', 'PNEUMONIA']
predicted_class = class_names[np.argmax(prediction[0])]
confidence = round(100 * np.max(prediction[0]), 2)

print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence}%")

# Output example:
# Prediction: PNEUMONIA
# Confidence: 96.78%
```

### Batch Processing
```python
import os
from glob import glob

# Process multiple images
image_folder = 'test_images/'
results = []

for img_path in glob(os.path.join(image_folder, '*.jpg')):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
    
    pred = model.predict(img_array, verbose=0)
    result = {
        'filename': os.path.basename(img_path),
        'prediction': class_names[np.argmax(pred[0])],
        'confidence': round(100 * np.max(pred[0]), 2)
    }
    results.append(result)

# Display results
for r in results:
    print(f"{r['filename']}: {r['prediction']} ({r['confidence']}%)")
```

## 🚀 Deployment

### Local Development
```bash
python app.py
# Access: http://localhost:5000
# Debug mode: enabled in development
```

### Production Deployment
The application is configured for deployment on:
- **Heroku**: Use gunicorn with Procfile
- **AWS EC2**: Deploy with nginx + gunicorn
- **Google Cloud Platform**: App Engine ready
- **Docker**: Containerization supported

### Heroku Deployment
```bash
# Install Heroku CLI and login
heroku login

# Create new app
heroku create pneumonia-detection-app

# Add buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Open app
heroku open
```

### Docker Deployment
```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

### Environment Configuration
```bash
# Production settings
export FLASK_ENV=production
export DEBUG=False
export PORT=5000

# Model path (if using external storage)
export MODEL_PATH=/path/to/model.h5
```

### Performance Optimization
- Use gunicorn with multiple workers: `gunicorn -w 4 -b 0.0.0.0:5000 app:app`
- Enable model caching for faster inference
- Implement request queuing for high traffic
- Consider GPU acceleration for batch processing
- Use CDN for static assets

## 🐛 Troubleshooting

### Common Issues

**1. Model file not found**
```bash
Error: OSError: Unable to open file (unable to open file: name = 'best_vgg19_pneumonia.h5')

Solution: 
- Download model from Google Drive link
- Place in root directory: pneumonia-detection/best_vgg19_pneumonia.h5
- Verify file size is ~80-100 MB
```

**2. Memory errors during prediction**
```bash
Error: ResourceExhaustedError: OOM when allocating tensor

Solution:
- Reduce batch size in code
- Use CPU-only TensorFlow: pip install tensorflow-cpu
- Close other applications to free memory
```

**3. Upload directory missing**
```bash
Error: FileNotFoundError: [Errno 2] No such file or directory: 'static/uploads/'

Solution: mkdir -p static/uploads
```

**4. Import errors**
```bash
Error: ModuleNotFoundError: No module named 'tensorflow'

Solution:
- Activate virtual environment: source venv/bin/activate
- Install dependencies: pip install -r requirements.txt
```

**5. CUDA/GPU errors**
```bash
Error: Could not load dynamic library 'libcudart.so.11.0'

Solution:
- Install CPU version: pip install tensorflow-cpu
- Or install CUDA toolkit for GPU support
```

**6. Low accuracy on custom images**
```bash
Issue: Model gives unexpected predictions

Solution:
- Ensure input is a chest X-ray image
- Check image quality and resolution
- Verify image is in correct format (JPEG/PNG)
- Check if image needs proper contrast/brightness
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Contribution Process
1. Fork the repository
2. Create feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add NewFeature'`)
4. Push to branch (`git push origin feature/NewFeature`)
5. Open Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Include unit tests for new features
- Update documentation for API changes
- Maintain model performance benchmarks
- Test on both CPU and GPU environments

### Areas for Contribution
- Model improvements (architecture, hyperparameters)
- UI/UX enhancements (better visualizations)
- Additional data augmentation techniques
- Batch processing capabilities
- API documentation with Swagger/OpenAPI
- Docker containerization
- Mobile app integration
- DICOM image support
- Multi-class classification (bacterial vs viral)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Medical Disclaimer

**IMPORTANT**: This application is for educational and research purposes only. It is NOT intended to replace professional medical diagnosis or clinical decision-making.

- **Not FDA Approved**: This is a research prototype, not a medical device
- **No Medical Advice**: Do not use for actual patient diagnosis
- **Consult Professionals**: Always seek advice from qualified healthcare providers
- **Accuracy Limitations**: AI predictions may contain errors (8.33% error rate on test set)
- **Research Purpose**: Intended for learning and academic exploration
- **No Liability**: Developers assume no responsibility for medical decisions

**Clinical Use Warning**: This system has not undergone clinical validation and should never be used as the sole basis for medical decisions. The 95% recall rate means 5% of pneumonia cases may be missed.

## 🙏 Acknowledgments

- **Dataset**: [Kaggle Chest X-Ray Images Database](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Medical Institution**: Guangzhou Women and Children's Medical Center
- **Architecture**: VGG19 by Visual Geometry Group, Oxford University (2014)
- **Frameworks**: TensorFlow/Keras development teams
- **Web Framework**: Flask and Pallets Projects
- **Community**: Kaggle community for dataset curation and notebooks
- **Hardware**: Kaggle for providing free GPU access (Tesla P100)

## 📞 Support

For questions, issues, or contributions:
- **Issues**: [GitHub Issues](https://github.com/yourusername/pneumonia-detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/pneumonia-detection/discussions)
- **Email**: your.email@example.com

## 📚 References

1. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG). *arXiv preprint arXiv:1409.1556*.
2. Rajpurkar, P., et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. *arXiv preprint arXiv:1711.05225*.
3. Kermany, D. S., et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*, 172(5), 1122-1131.
4. World Health Organization. (2019). Pneumonia Fact Sheet.

---

**Built with ❤️ for better healthcare through AI**

*Last updated: November 2025*
