# 🐟 Multiclass Fish Image Classification

### Sharable **Drive link**: https://drive.google.com/drive/folders/1j78PgmWQZf6Quqj_j4G2yjAPn9UebL9B?usp=sharing

### **StreamLit Live app link**: 
## 📌 Overview
This project focuses on building, comparing, and deploying deep learning models to classify images of multiple fish species.  
We train:
- **1 Custom CNN (built from scratch)**
- **5 Pre-trained models with Transfer Learning**:
  - VGG16
  - ResNet50
  - MobileNetV2
  - InceptionV3
  - EfficientNetB0

The best-performing model is deployed as a simple **Streamlit web application** for real-time predictions.

---

## 📂 Dataset
**Source:** Provided dataset stored in Google Drive.  
**Structure:**

Multi Fish Image classification/
│
└── data/
├── train/
│ ├── animal fish/
│ ├── animal fish bass/
│ ├── fish sea_food black_sea_sprat/
│ ├── ...
├── val/
│ ├── animal fish/
│ ├── ...
├── test/
├── animal fish/
├── ...

- Train / Validation / Test splits are already provided.
- Each class folder contains images of that fish species.
- Total size: ~258 MB.

---

## 🎯 Objectives
1. **Data Preparation**: Load and preprocess images using `ImageDataGenerator` for training, validation, and testing.
2. **EDA**: Visualize class distributions, sample images, and check image dimensions/format.
3. **Model Training**:
   - Build and train a CNN from scratch.
   - Fine-tune 5 transfer learning models.
4. **Evaluation**: Compare performance using metrics like accuracy, precision, recall, F1-score, confusion matrix.
5. **Deployment**: Use Streamlit for interactive classification.

---

## 🛠️ Tech Stack
- **Python 3.x**
- **TensorFlow / Keras**
- **NumPy, Pandas, Matplotlib, Seaborn**
- **OpenCV, Pillow**
- **scikit-learn**
- **Streamlit** (for deployment)
- **Google Colab** (for model training)

---

## 📜 Project Workflow
### Step 1: Project Setup & Dataset Loading
- Mount Google Drive in Colab.
- Load train, validation, test sets using `flow_from_directory`.
- View sample images.

### Step 2: Understanding Variables
- Create a DataFrame with `image_path`, `class_label`, `split`.
- Summarize dataset statistics and unique values.

### Step 3: Data Visualization
- Class distribution charts, sample images per class.
- Pie and bar charts for storytelling.

### Step 4: Image & Dimension Check
- Verify all images are valid and properly formatted.
- Check color channels and consistent dimensions.

### Step 5: Model Building & Training
- Custom CNN architecture.
- VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0 fine-tuned.

### Step 6: Model Evaluation
- Plot training/validation accuracy & loss.
- Generate classification reports & confusion matrices.

### Step 7: Model Comparison
- Tabulate & visualize metrics of all six models.

### Step 8: Model Saving
- Save models in `.h5` or `.keras` format.

### Step 9: Deployment Preparation
- Export the best model.
- Write a Streamlit script for live classification.

### Step 10: Documentation
- Maintain README, code comments, and usage guide.

---

## 🚀 Running the Project
**1. Clone the repo & open in Google Colab**
git clone <your-repo-url>

 Alternatively, you can upload the `.ipynb` notebook to your Google Drive and open it in Colab.

**2. Mount Google Drive**
from google.colab import drive
drive.mount('/content/drive')


**3. Update Dataset Paths**
train_path = '/content/drive/MyDrive/Multi Fish Image classification/data/train'
valid_path = '/content/drive/MyDrive/Multi Fish Image classification/data/val'
test_path = '/content/drive/MyDrive/Multi Fish Image classification/data/test'


**4. Install Dependencies (if needed)**
pip install tensorflow seaborn pillow opencv-python streamlit


**5. Run all notebook cells**
Train all six models and evaluate them.

**6. Deployment (Streamlit)**
streamlit run app.py


---

## 📊 Results
A sample performance comparison table:

| Model          | Accuracy | Precision | Recall | F1-score |
|----------------|----------|-----------|--------|----------|
| CNN (scratch)  | 84%      | ...       | ...    | ...      |
| VGG16          | 92%      | ...       | ...    | ...      |
| ResNet50       | 94%      | ...       | ...    | ...      |
| MobileNetV2    | 91%      | ...       | ...    | ...      |
| InceptionV3    | 93%      | ...       | ...    | ...      |
| EfficientNetB0 | 92%      | ...       | ...    | ...      |


---

## 📌 Deliverables
- Trained model files (`.h5` / `.keras`)
- Google Colab notebook with code & markdown summaries.
- Model comparison report.
- Streamlit application for deployment.
- README documentation.

---

## 📄 License
This project is for educational purposes. Modify and adapt freely.
Author: Siddhartha Ram Konasani
mail: siddharthakonasani.77@gmail.com
