import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    return stopwords.words('english')

stop_words = download_nltk_data()

# Page configuration
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .spam-badge {
        background-color: #ff4444;
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-flex;
        align-items: center;
        gap: 10px;
    }
    .ham-badge {
        background-color: #00C851;
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-flex;
        align-items: center;
        gap: 10px;
    }
    .icon-img {
        width: 30px;
        height: 30px;
        vertical-align: middle;
    }
    </style>
""", unsafe_allow_html=True)

class SpamDetector:
    """Spam detection inference class"""
    
    def __init__(self, model_path, vectorizer_path):
        self.model = self.load_model(model_path)
        self.vectorizer = self.load_vectorizer(vectorizer_path)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stop_words)
    
    @staticmethod
    @st.cache_resource
    def load_model(model_path):
        """Load trained model"""
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    @st.cache_resource
    def load_vectorizer(vectorizer_path):
        """Load vectorizer"""
        with open(vectorizer_path, 'rb') as f:
            return pickle.load(f)
    
    def preprocess_text(self, text):
        """Clean and preprocess email text"""
        if pd.isna(text) or text == "":
            return ""
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        words = text.split()
        words = [self.stemmer.stem(word) for word in words
                 if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def predict(self, text):
        """Predict if email is spam"""
        processed = self.preprocess_text(text)
        
        if not processed:
            return {
                'is_spam': False,
                'confidence': 0.5,
                'spam_probability': 0.5,
                'ham_probability': 0.5,
                'error': 'Text is empty after preprocessing'
            }
        
        vectorized = self.vectorizer.transform([processed])
        prediction = self.model.predict(vectorized)[0]
        probability = self.model.predict_proba(vectorized)[0]
        
        return {
            'is_spam': bool(prediction),
            'confidence': float(probability[prediction]),
            'spam_probability': float(probability[1]),
            'ham_probability': float(probability[0]),
            'processed_text': processed
        }

def create_gauge_chart(value, title, color):
    """Create a gauge chart for probability visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24, 'color': '#333'}},
        number={'suffix': "%", 'font': {'size': 50, 'color': color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#666"},
            'bar': {'color': color, 'thickness': 0.7},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ddd",
            'steps': [
                {'range': [0, 50], 'color': '#e8f5e9'},
                {'range': [50, 75], 'color': '#fff3e0'},
                {'range': [75, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=30, r=30, t=80, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#333", 'family': "Arial"}
    )
    
    return fig

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">Email Spam Detector</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        st.info("This system uses an ensemble of ML models (Naive Bayes, SVM, Random Forest) to detect spam emails with high accuracy.")
        
        st.header("Model Info")
        model_dir = Path('models')
        
        if model_dir.exists():
            model_path = model_dir / 'spam_detector_model.pkl'
            vectorizer_path = model_dir / 'vectorizer.pkl'
            
            if model_path.exists() and vectorizer_path.exists():
                st.success("Model loaded successfully")
                st.caption(f"Model: {model_path.name}")
                st.caption(f"Vectorizer: {vectorizer_path.name}")
            else:
                st.error("Model files not found")
                st.info("Please train the model first using the training script.")
        else:
            st.error("Models directory not found")
        
        st.header("How to Use")
        st.markdown("""
        1. Enter or paste email text
        2. Upload a CSV file for batch processing
        3. Click Analyze to get results
        4. View confidence scores and classifications
        """)
        
        st.header("Features")
        st.markdown("""
        - Real-time spam detection
        - Confidence score display
        - Batch processing support
        - Visual probability gauges
        - Text preprocessing insights
        """)
    
    # Check if model exists
    model_dir = Path('models')
    model_path = model_dir / 'spam_detector_model.pkl'
    vectorizer_path = model_dir / 'vectorizer.pkl'
    
    if not (model_path.exists() and vectorizer_path.exists()):
        st.error("Model not found! Please train the model first using the training script.")
        st.code("""# Run the training script first:
python train_spam_detector.py
        """)
        return
    
    # Load detector
    detector = SpamDetector(model_path, vectorizer_path)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Single Email", "Batch Upload", "About"])
    
    with tab1:
        st.header("Analyze Single Email")
        
        # Sample emails
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load Legitimate Email Sample"):
                st.session_state.email_text = "Hi John, let's meet for coffee tomorrow at 3pm. Looking forward to catching up! Best regards, Sarah"
        with col2:
            if st.button("Load Spam Email Sample"):
                st.session_state.email_text = "CONGRATULATIONS!!! You've WON $1,000,000! Click here NOW to claim your prize!!! Act fast, offer expires soon!!!"
        
        # Email input
        email_text = st.text_area(
            "Enter Email Text:",
            value=st.session_state.get('email_text', ''),
            height=200,
            placeholder="Paste your email content here..."
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            analyze_button = st.button("Analyze Email", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("Clear", use_container_width=True)
        
        if clear_button:
            st.session_state.email_text = ''
            st.rerun()
        
        if analyze_button and email_text:
            with st.spinner("Analyzing email..."):
                result = detector.predict(email_text)
            
            st.markdown("---")
            st.header("Analysis Results")
            
            # Display result badge with icons
            if result['is_spam']:
                st.markdown('''<div class="spam-badge">
                    <img src="https://img.icons8.com/?size=100&id=37971&format=png&color=000000" class="icon-img">
                    SPAM DETECTED
                </div>''', unsafe_allow_html=True)
            else:
                st.markdown('''<div class="ham-badge">
                    <img src="https://img.icons8.com/?size=100&id=124383&format=png&color=000000" class="icon-img">
                    LEGITIMATE EMAIL
                </div>''', unsafe_allow_html=True)
            
            st.markdown("")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Confidence", f"{result['confidence']:.1%}")
            with col2:
                st.metric("Spam Probability", f"{result['spam_probability']:.1%}")
            with col3:
                st.metric("Ham Probability", f"{result['ham_probability']:.1%}")
            
            # Gauge charts
            st.subheader("Probability Visualization")
            col1, col2 = st.columns(2)
            
            with col1:
                spam_gauge = create_gauge_chart(result['spam_probability'], "Spam Score", "#ffffff")
                st.plotly_chart(spam_gauge, use_container_width=True)
            
            with col2:
                ham_gauge = create_gauge_chart(result['ham_probability'], "Legitimate Score", "#ffffff")
                st.plotly_chart(ham_gauge, use_container_width=True)
            
            # Processed text
            with st.expander("View Preprocessed Text"):
                st.text(result.get('processed_text', 'N/A'))
            
            # Risk assessment
            st.subheader("Risk Assessment")
            if result['spam_probability'] >= 0.9:
                st.error("HIGH RISK: This email is very likely spam. Do not interact with it.")
            elif result['spam_probability'] >= 0.7:
                st.warning("MEDIUM RISK: This email shows spam characteristics. Be cautious.")
            elif result['spam_probability'] >= 0.5:
                st.info("LOW RISK: This email appears mostly legitimate, but verify sender.")
            else:
                st.success("SAFE: This email appears to be legitimate.")
    
    with tab2:
        st.header("Batch Email Analysis")
        st.markdown("Upload a CSV file with an 'email' or 'text' column to analyze multiple emails at once.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.subheader("Preview")
            st.dataframe(df.head())
            
            # Identify text column
            text_col = None
            for col in ['text', 'email', 'message', 'content', 'body', 'Text', 'Email', 'Message']:
                if col in df.columns:
                    text_col = col
                    break
            
            if text_col is None:
                st.error("Could not find a text column. Please ensure your CSV has a column named 'text', 'email', or 'message'.")
            else:
                if st.button("Analyze All Emails", type="primary"):
                    with st.spinner(f"Analyzing {len(df)} emails..."):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for idx, text in enumerate(df[text_col]):
                            result = detector.predict(str(text))
                            results.append({
                                'Email': str(text)[:100] + '...',
                                'Classification': 'SPAM' if result['is_spam'] else 'HAM',
                                'Confidence': f"{result['confidence']:.1%}",
                                'Spam Probability': f"{result['spam_probability']:.1%}"
                            })
                            progress_bar.progress((idx + 1) / len(df))
                        
                        results_df = pd.DataFrame(results)
                        
                        st.success(f"Analyzed {len(results)} emails!")
                        
                        # Summary metrics
                        spam_count = sum(1 for r in results if r['Classification'] == 'SPAM')
                        ham_count = len(results) - spam_count
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Emails", len(results))
                        with col2:
                            st.metric("Spam Detected", spam_count)
                        with col3:
                            st.metric("Legitimate Emails", ham_count)
                        
                        # Pie chart
                        fig = px.pie(
                            values=[spam_count, ham_count],
                            names=['Spam', 'Legitimate'],
                            title='Email Classification Distribution',
                            color_discrete_sequence=['#ff4444', '#00C851']
                        )
                        st.plotly_chart(fig)
                        
                        # Results table
                        st.subheader("Detailed Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name="spam_detection_results.csv",
                            mime="text/csv"
                        )
    
    with tab3:
        st.header("About This System")
        
        st.markdown("""
        ### Objective
        This NLP-based system detects and filters spam or phishing emails using advanced text classification techniques.
        
        ### Technology Stack
        - **Machine Learning Models**: Ensemble of Naive Bayes, SVM, and Random Forest
        - **NLP Techniques**: TF-IDF vectorization, text preprocessing, stemming
        - **Framework**: Streamlit for interactive web interface
        - **Libraries**: scikit-learn, NLTK, pandas, plotly
        
        ### Features
        - **Automatic Detection**: Real-time spam classification
        - **Confidence Scores**: Probability metrics for predictions
        - **Batch Processing**: Analyze multiple emails at once
        - **Visual Analytics**: Interactive gauges and charts
        - **Text Preprocessing**: Advanced NLP pipeline
        
        ### How It Works
        1. **Text Preprocessing**: Removes URLs, emails, special characters
        2. **Tokenization**: Splits text into words
        3. **Stemming**: Reduces words to root form
        4. **Vectorization**: Converts text to numerical features (TF-IDF)
        5. **Classification**: Ensemble model predicts spam/ham
        6. **Confidence**: Returns probability scores
        
        ### Model Performance
        The ensemble model combines multiple algorithms to achieve high accuracy:
        - Naive Bayes: Fast and effective for text classification
        - SVM: Strong performance on high-dimensional data
        - Random Forest: Robust ensemble method
        
        ### Security Note
        This tool helps identify potentially malicious emails but should be used as part of a comprehensive security strategy.
        Always verify suspicious emails through other means.
        """)
        
        st.info("Tip: For best results, train the model on a large, diverse dataset of emails.")

if __name__ == "__main__":
    main()