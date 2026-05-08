import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# NLTK setup
nltk.download('stopwords')

port_stem = PorterStemmer()

# Text preprocessing
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()

    stemmed_content = [
        port_stem.stem(word)
        for word in stemmed_content
        if not word in stopwords.words('english')
    ]

    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content

# Streamlit UI
st.title("📰 Fake News Detection System")

input_data = st.text_area("Enter News Headline")

if st.button("Check News"):

    if input_data.strip() == "":
        st.warning("Please enter some news text.")
    else:
        processed_data = stemming(input_data)

        vector_input = vectorizer.transform([processed_data])

        prediction = model.predict(vector_input)

        if prediction[0] == 0:
            st.error("🚨 Fake News Detected")
        else:
            st.success("✅ Real News")