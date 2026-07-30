import streamlit as st
import joblib
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
st.set_page_config(page_title="Sentiment Analyzer", page_icon="📊", layout="centered")


# NLTK setup
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text cleaning function (Same as Jupyter)
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(cleaned_words)

# Model aur Vectorizer load karo
@st.cache_resource
def load_artifacts():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_artifacts()

# Streamlit UI Design

st.title("📊 Sentiment Analysis Web App")
st.write("Enter any text/review below to predict its sentiment using Machine Learning!")

# User input text area
user_input = st.text_area("Write your review or text here:", height=150)

if st.button("Analyze Sentiment", type="primary"):
    if user_input.strip() != "":
        # Preprocessing & Prediction
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]
        
        st.subheader("Result:")
        
        # Display badge based on result
        if prediction == "positive":
            st.success(f"**Sentiment:** POSITIVE 😊")
        elif prediction == "negative":
            st.error(f"**Sentiment:** NEGATIVE 😡")
        else:
            st.warning(f"**Sentiment:** NEUTRAL 😐")
            
        # Optional: Confidence score / Probabilities
        st.write("---")
        st.write("**Prediction Probabilities:**")
        classes = model.classes_
        for cls, prob in zip(classes, probs):
            st.write(f"- {cls.capitalize()}: {prob*100:.2f}%")
            st.progress(float(prob))
            
    else:
        st.warning("Please enter some text first!")