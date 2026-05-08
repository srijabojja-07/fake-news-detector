# Fake News Detector

An AI-powered Fake News Detection System built using Machine Learning, NLP, and Streamlit.

## Features
- Detects whether news is REAL or FAKE
- Machine Learning based prediction
- Streamlit web interface
- NLP preprocessing using TF-IDF

## Technologies Used
- Python
- Scikit-learn
- Pandas
- NLTK
- Streamlit

## Project Structure

fake_news_detector/
│
├── dataset/
├── model/
├── app.py
├── train_model.py
├── requirements.txt

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Accuracy
Achieved approximately 94% accuracy using Logistic Regression and TF-IDF Vectorization.
