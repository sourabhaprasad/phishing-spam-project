import pandas as pd
import numpy as np
import re
import pickle
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Import configuration
try:
    from config import DATASET_PATH, MODEL_DIR, MODEL_FILENAME, VECTORIZER_FILENAME
    from config import TEST_SIZE, RANDOM_STATE, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE
    from config import TFIDF_MIN_DF, TFIDF_MAX_DF, NAIVE_BAYES_ALPHA, SVM_C, SVM_KERNEL
    from config import RANDOM_FOREST_N_ESTIMATORS
except ImportError:
    print("⚠️  Warning: config.py not found. Using default settings.")
    DATASET_PATH = "/Users/sourabha/.cache/kagglehub/datasets/meruvulikith/190k-spam-ham-email-dataset-for-classification/versions/1"
    MODEL_DIR = "models"
    MODEL_FILENAME = "spam_detector_model.pkl"
    VECTORIZER_FILENAME = "vectorizer.pkl"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    TFIDF_MAX_FEATURES = 5000
    TFIDF_NGRAM_RANGE = (1, 2)
    TFIDF_MIN_DF = 2
    TFIDF_MAX_DF = 0.95
    NAIVE_BAYES_ALPHA = 0.1
    SVM_C = 1.0
    SVM_KERNEL = 'linear'
    RANDOM_FOREST_N_ESTIMATORS = 100

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SpamDetectionPipeline:
    """Complete pipeline for spam email detection"""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.vectorizer = None
        self.model = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self):
        """Load and explore the dataset"""
        print("Loading dataset...")
        
        # Try to find CSV files in the dataset directory
        data_dir = Path(self.dataset_path)
        csv_files = list(data_dir.glob('*.csv'))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.dataset_path}")
        
        # Load the first CSV file
        df = pd.read_csv(csv_files[0])
        print(f"\nDataset loaded from: {csv_files[0]}")
        print(f"Shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nFirst few rows:\n{df.head()}")
        
        return df
    
    def preprocess_text(self, text):
        """Clean and preprocess email text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        words = text.split()
        words = [self.stemmer.stem(word) for word in words 
                if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def prepare_data(self, df):
        """Prepare data for training"""
        print("\nPreparing data...")
        
        # Identify text and label columns
        # Common column names for text
        text_col = None
        for col in ['text', 'message', 'email', 'content', 'body', 'Text', 'Message', 'Email']:
            if col in df.columns:
                text_col = col
                break
        
        # Common column names for labels
        label_col = None
        for col in ['label', 'spam', 'category', 'class', 'Label', 'Category']:
            if col in df.columns:
                label_col = col
                break
        
        if text_col is None or label_col is None:
            print("\nAvailable columns:", df.columns.tolist())
            raise ValueError("Could not identify text and label columns")
        
        print(f"Using text column: {text_col}")
        print(f"Using label column: {label_col}")
        
        # Extract text and labels
        texts = df[text_col].values
        labels = df[label_col].values
        
        # Convert labels to binary (0: ham, 1: spam)
        unique_labels = np.unique(labels)
        print(f"\nUnique labels: {unique_labels}")
        
        # Map labels to binary
        if len(unique_labels) == 2:
            # Identify which label is spam
            spam_label = None
            for label in unique_labels:
                if str(label).lower() in ['spam', '1', 'true']:
                    spam_label = label
                    break
            
            if spam_label is None:
                spam_label = unique_labels[1]  # Assume second label is spam
            
            labels = (labels == spam_label).astype(int)
        
        print(f"\nLabel distribution:")
        print(f"Ham (0): {np.sum(labels == 0)}")
        print(f"Spam (1): {np.sum(labels == 1)}")
        
        return texts, labels
    
    def train_model(self, texts, labels):
        """Train the spam detection model"""
        print("\n" + "="*60)
        print("TRAINING SPAM DETECTION MODEL")
        print("="*60)
        
        # Preprocess texts
        print("\nPreprocessing texts...")
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Vectorization
        print("\nVectorizing texts...")
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF
        )
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"Feature vector shape: {X_train_vec.shape}")
        
        # Train multiple models
        print("\n" + "-"*60)
        print("Training Naive Bayes model...")
        nb_model = MultinomialNB(alpha=NAIVE_BAYES_ALPHA)
        nb_model.fit(X_train_vec, y_train)
        nb_pred = nb_model.predict(X_test_vec)
        nb_acc = accuracy_score(y_test, nb_pred)
        print(f"Naive Bayes Accuracy: {nb_acc:.4f}")
        
        print("\nTraining SVM model...")
        base_svm = LinearSVC(C=SVM_C, random_state=RANDOM_STATE)
        svm_model = CalibratedClassifierCV(base_svm)   # gives predict_proba() method
        svm_model.fit(X_train_vec, y_train)
        svm_pred = svm_model.predict(X_test_vec)
        svm_acc = accuracy_score(y_test, svm_pred)
        print(f"SVM Accuracy: {svm_acc:.4f}")
        
        print("\nTraining Random Forest model...")
        rf_model = RandomForestClassifier(n_estimators=RANDOM_FOREST_N_ESTIMATORS, 
                                         random_state=RANDOM_STATE, n_jobs=-1)
        rf_model.fit(X_train_vec, y_train)
        rf_pred = rf_model.predict(X_test_vec)
        rf_acc = accuracy_score(y_test, rf_pred)
        print(f"Random Forest Accuracy: {rf_acc:.4f}")
        
        # Ensemble model
        print("\nTraining Ensemble model...")
        self.model = VotingClassifier(
            estimators=[
                ('nb', nb_model),
                ('svm', svm_model),
                ('rf', rf_model)
            ],
            voting='soft'
        )
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_vec)
        y_pred_proba = self.model.predict_proba(X_test_vec)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*60)
        print("FINAL MODEL PERFORMANCE")
        print("="*60)
        print(f"\nEnsemble Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"True Negatives (Ham): {cm[0][0]}")
        print(f"False Positives (Ham as Spam): {cm[0][1]}")
        print(f"False Negatives (Spam as Ham): {cm[1][0]}")
        print(f"True Positives (Spam): {cm[1][1]}")
        
        return accuracy
    
    def save_model(self, model_dir=None):
        """Save trained model and vectorizer"""
        if model_dir is None:
            model_dir = MODEL_DIR
            
        Path(model_dir).mkdir(exist_ok=True)
        
        model_path = Path(model_dir) / MODEL_FILENAME
        vectorizer_path = Path(model_dir) / VECTORIZER_FILENAME
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"\n✓ Model saved to: {model_path}")
        print(f"✓ Vectorizer saved to: {vectorizer_path}")
    
    def predict(self, text):
        """Predict if email is spam"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Preprocess
        processed = self.preprocess_text(text)
        
        # Vectorize
        vectorized = self.vectorizer.transform([processed])
        
        # Predict
        prediction = self.model.predict(vectorized)[0]
        probability = self.model.predict_proba(vectorized)[0]
        
        result = {
            'is_spam': bool(prediction),
            'confidence': float(probability[prediction]),
            'spam_probability': float(probability[1]),
            'ham_probability': float(probability[0])
        }
        
        return result


def main():
    """Main training pipeline"""
    
    print("="*60)
    print("EMAIL SPAM DETECTION - TRAINING PIPELINE")
    print("="*60)
    print(f"\n📁 Dataset Path: {DATASET_PATH}")
    print(f"💾 Model Directory: {MODEL_DIR}")
    print(f"🎯 Test Size: {TEST_SIZE}")
    print(f"🔢 Random State: {RANDOM_STATE}")
    print("="*60)
    
    # Initialize pipeline with dataset path from config
    pipeline = SpamDetectionPipeline(DATASET_PATH)
    
    # Load data
    df = pipeline.load_data()
    
    # Prepare data
    texts, labels = pipeline.prepare_data(df)
    
    # Train model
    accuracy = pipeline.train_model(texts, labels)
    
    # Save model
    pipeline.save_model()
    
    # Test with sample emails
    print("\n" + "="*60)
    print("TESTING WITH SAMPLE EMAILS")
    print("="*60)
    
    test_emails = [
        "Hi John, let's meet for coffee tomorrow at 3pm. Looking forward to catching up!",
        "CONGRATULATIONS!!! You've WON $1,000,000! Click here NOW to claim your prize!!!",
        "The meeting has been rescheduled to next Monday. Please confirm your attendance.",
        "FREE VIAGRA!!! Best prices online. No prescription needed. Order now!"
    ]
    
    for i, email in enumerate(test_emails, 1):
        result = pipeline.predict(email)
        print(f"\nEmail {i}:")
        print(f"Text: {email[:80]}...")
        print(f"Prediction: {'SPAM' if result['is_spam'] else 'HAM'}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Spam Probability: {result['spam_probability']:.2%}")


if __name__ == "__main__":
    main()