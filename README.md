# Email Spam Detection System

## Table of Contents

1. [Requirements](#requirements)
2. [Installation Steps](#installation-steps)
3. [Dataset Setup](#dataset-setup)
4. [Training the Model](#training-the-model)
5. [Launching the Web Application](#launching-the-web-application)
6. [Project Structure](#project-structure)
7. [Usage](#usage)
8. [Model Features](#model-features)
9. [Customization](#customization)
10. [Performance Metrics](#performance-metrics)
11. [Troubleshooting](#troubleshooting)
12. [Security Considerations](#security-considerations)
13. [Additional Resources](#additional-resources)
14. [Contributing](#contributing)
15. [License](#license)

---

## Requirements

- Python 3.11+
- Libraries: listed in `requirements.txt`:

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
nltk>=3.8.0
streamlit>=1.28.0
plotly>=5.17.0
kagglehub>=0.2.0
```

---

## Installation Steps

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv spam_detector_env

# Activate virtual environment
# On Windows
spam_detector_env\Scripts\activate
# On Mac/Linux
source spam_detector_env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Create `download_dataset.py`:

```python
import kagglehub

# Download dataset
path = kagglehub.dataset_download("meruvulikith/190k-spam-ham-email-dataset-for-classification")
print("Path to dataset files:", path)
```

Run:

```bash
python download_dataset.py
```

---

## Dataset Setup

You have two options to specify the dataset path:

1. **Using `config.py` (recommended)**

```python
# config.py
DATASET_PATH = "/Users/sourabha/.cache/kagglehub/datasets/meruvulikith/190k-spam-ham-email-dataset-for-classification/versions/1"
```

- Use the **directory path**, not the full CSV path.
- The script will automatically locate the CSV inside this folder.

2. **Verify your dataset path**

```bash
# Mac/Linux
ls -la /Users/sourabha/.cache/kagglehub/datasets/meruvulikith/190k-spam-ham-email-dataset-for-classification/versions/1/

# Windows
dir C:\Users\sourabha\.cache\kagglehub\datasets\meruvulikith\190k-spam-ham-email-dataset-for-classification\versions\1\
```

You should see `spam_Emails_data.csv`.

---

## Training the Model

Update the dataset path in `train_spam_detector.py` if needed, then run:

```bash
python train_spam_detector.py
```

**Training Output Example:**

```
EMAIL SPAM DETECTION - TRAINING PIPELINE
Dataset Path: /Users/sourabha/.cache/kagglehub/...
Model Directory: models
Test Size: 0.2
Random State: 42

Dataset loaded from: spam_Emails_data.csv
Shape: (193852, 2)
Columns: ['label', 'text']

Training Naive Bayes model...
Naive Bayes Accuracy: 0.9289

Training SVM model...
SVM Accuracy: 0.9746

Training Random Forest model...
Random Forest Accuracy: 0.9863

Training Ensemble model...
Ensemble Model Accuracy: 0.9811

Confusion Matrix:
True Negatives (Ham): 20028
False Positives (Ham as Spam): 404
False Negatives (Spam as Ham): 327
True Positives (Spam): 18012

Model saved to: models/spam_detector_model.pkl
Vectorizer saved to: models/vectorizer.pkl
```

- Training may take 5–15 minutes depending on your machine.

---

## Launching the Web Application

```bash
streamlit run streamlit.py
```

The app will open at `http://localhost:8501`.

---

## Project Structure

```
spam-email-detector/
├── config.py                   # Dataset path
├── train_spam_detector.py      # Training script
├── streamlit.py                # Streamlit web app
├── requirements.txt            # Dependencies
├── download_dataset.py         # Dataset downloader
├── models/                     # Saved models (created after training)
│   ├── spam_detector_model.pkl
│   └── vectorizer.pkl
└── README.md                   # Documentation
```

---

## Usage

### Single Email Analysis

1. Open the web app
2. Go to "Single Email" tab
3. Enter or paste email text
4. Click "Analyze Email"
5. View results: classification, confidence scores, risk assessment

### Batch Processing

1. Go to "Batch Upload" tab
2. Upload a CSV file containing an email column
3. Click "Analyze All Emails"
4. View summary statistics and download results

---

## Model Features

**Text Preprocessing:**

- Lowercase conversion
- URL removal
- Email address removal
- Special character removal
- Stopword removal
- Stemming (Porter Stemmer)

**Vectorization:**

- TF-IDF
- Max features: 5000
- N-grams: unigrams and bigrams
- Min document frequency: 2
- Max document frequency: 95%

**Classification Models:**

- Multinomial Naive Bayes
- Support Vector Machine (Linear kernel)
- Random Forest
- Voting Classifier (ensemble of all three)

---

## Customization

- Adjust vectorizer parameters in `train_spam_detector.py`
- Adjust model parameters (e.g., `alpha`, `C`, `n_estimators`)
- Extend preprocessing in `preprocess_text()`

---

## Performance Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Troubleshooting

- **NLTK Data Not Found**:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

- **Model Files Not Found**: Ensure `train_spam_detector.py` was run and `models/` contains both `.pkl` files.

- **CSV Upload Error**: Ensure the CSV contains one of the columns: `text`, `email`, `message`, `content`, `body`.

- **Memory Error During Training**: Reduce `max_features` or dataset size.

---

## Additional Resources

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [NLTK Documentation](https://www.nltk.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- Text Classification Tutorials
