import pandas as pd
import numpy as np
import nltk
import re
import joblib

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

# Load datasets
fake_data = pd.read_csv("dataset/Fake.csv")
true_data = pd.read_csv("dataset/True.csv")

# Add labels
fake_data["label"] = 0
true_data["label"] = 1

# Combine datasets
data = pd.concat([fake_data, true_data], axis=0)

# Shuffle dataset
data = data.sample(frac=1)

# Reset index
data.reset_index(inplace=True)

# Use title column
data = data[["title", "label"]]

# Stemming function
port_stem = PorterStemmer()

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

# Apply stemming
data['title'] = data['title'].apply(stemming)

# Split data
X = data['title']
Y = data['label']

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# Model training
model = LogisticRegression()
model.fit(X_train, Y_train)

# Accuracy
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print("Accuracy Score:", testing_data_accuracy)

# Save model and vectorizer
joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model saved successfully!")