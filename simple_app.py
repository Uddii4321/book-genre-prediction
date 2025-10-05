import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Simple text preprocessing function"""
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    processed_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 2:
            lemmatized = lemmatizer.lemmatize(token)
            processed_tokens.append(lemmatized)
    
    return ' '.join(processed_tokens)

def main():
    st.set_page_config(
        page_title="Book Genre Classification - Simple Version",
        page_icon="📚",
        layout="wide"
    )
    
    # Title and description
    st.title("📚 Book Genre Classification - Simple Version")
    st.markdown("""
    This is a simplified version of the book genre classification app.
    Enter a book title and description below to get started!
    """)
    
    # Check if model files exist
    try:
        model = joblib.load('model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        st.success("✅ Models loaded successfully!")
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.info("Please run the Jupyter notebook first to train and save the models.")
        return
    
    # Create two columns for input and output
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        # Input fields
        title = st.text_input(
            "Book Title",
            placeholder="Enter the book title...",
            help="Enter the title of the book you want to classify"
        )
        
        description = st.text_area(
            "Book Description",
            placeholder="Enter the book description...",
            help="Enter a detailed description of the book",
            height=150
        )
        
        # Predict button
        predict_button = st.button("🔮 Predict Genre", type="primary")
    
    with col2:
        st.header("🎯 Prediction Results")
        
        if predict_button:
            if title.strip() and description.strip():
                try:
                    # Combine title and description
                    combined_text = f"{title} {description}"
                    
                    # Preprocess the text
                    cleaned_text = preprocess_text(combined_text)
                    
                    # Vectorize the text
                    text_vector = vectorizer.transform([cleaned_text])
                    
                    # Make prediction
                    prediction = model.predict(text_vector)[0]
                    probability = model.predict_proba(text_vector)[0]
                    
                    # Get the predicted genre
                    predicted_genre = label_encoder.inverse_transform([prediction])[0]
                    
                    # Get confidence score
                    confidence = np.max(probability)
                    
                    # Display main prediction
                    st.success(f"**Predicted Genre: {predicted_genre}**")
                    
                    # Confidence score
                    confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
                    st.markdown(f"**Confidence:** :{confidence_color}[{confidence:.2%}]")
                    
                    # Progress bar for confidence
                    st.progress(confidence)
                    
                    # Get top 3 predictions
                    top_3_indices = np.argsort(probability)[-3:][::-1]
                    top_3_genres = label_encoder.inverse_transform(top_3_indices)
                    top_3_scores = probability[top_3_indices]
                    
                    # Top 3 predictions
                    st.subheader("🏆 Top 3 Predictions")
                    for i, (genre, score) in enumerate(zip(top_3_genres, top_3_scores)):
                        st.write(f"{i+1}. **{genre}** - {score:.2%}")
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                
            else:
                st.warning("Please enter both a title and description to get a prediction.")
    
    # Sidebar with additional information
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown("""
        **Model Details:**
        - Algorithm: Logistic Regression
        - Features: TF-IDF vectorization
        - Text Processing: NLTK preprocessing pipeline
        - Genres Supported: 10 different book genres
        """)
        
        st.header("🎯 Supported Genres")
        if 'label_encoder' in locals():
            genres = label_encoder.classes_
            for i, genre in enumerate(genres):
                st.write(f"{i+1}. {genre}")

if __name__ == "__main__":
    main()
