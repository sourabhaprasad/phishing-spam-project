"""
Configuration file for Spam Email Detection System
Update the paths below to match your local setup
"""

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

# Option 1: Use the full directory path (recommended)
# The script will automatically find CSV files in this directory
DATASET_PATH = "/Users/sourabha/.cache/kagglehub/datasets/meruvulikith/190k-spam-ham-email-dataset-for-classification/versions/1"

# Option 2: Or use the full CSV file path directly
# DATASET_CSV_FILE = "/Users/sourabha/.cache/kagglehub/datasets/meruvulikith/190k-spam-ham-email-dataset-for-classification/versions/1/spam_Emails_data.csv"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Directory where trained models will be saved
MODEL_DIR = "models"

# Model filenames
MODEL_FILENAME = "spam_detector_model.pkl"
VECTORIZER_FILENAME = "vectorizer.pkl"

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Train-test split ratio
TEST_SIZE = 0.2

# Random seed for reproducibility
RANDOM_STATE = 42

# Vectorizer settings
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)  # Use unigrams and bigrams
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95

# Model hyperparameters
NAIVE_BAYES_ALPHA = 0.1
SVM_C = 1.0
SVM_KERNEL = 'linear'
RANDOM_FOREST_N_ESTIMATORS = 100

# =============================================================================
# STREAMLIT APP CONFIGURATION
# =============================================================================

# Page configuration
PAGE_TITLE = "Email Spam Detector"
PAGE_ICON = "📧"
LAYOUT = "wide"

# Sample emails for testing
SAMPLE_LEGITIMATE_EMAIL = """Hi John, 

I hope this email finds you well. I wanted to follow up on our meeting last week regarding the Q4 project timeline. 

Could we schedule a brief call tomorrow at 3pm to discuss next steps?

Looking forward to hearing from you.

Best regards,
Sarah"""

SAMPLE_SPAM_EMAIL = """CONGRATULATIONS!!! 

You've been selected as our LUCKY WINNER! You've WON $1,000,000 in cash prizes!!!

Click here NOW to claim your prize before it expires: http://fake-link.com

This is a LIMITED TIME OFFER! ACT FAST!!!

Reply with your bank details to receive your winnings immediately!

FREE! FREE! FREE! No purchase necessary!"""