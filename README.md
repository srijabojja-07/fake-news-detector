# 📰 Fake News Detection System

An AI-powered Fake News Detection System developed using Machine Learning, Natural Language Processing (NLP), and Streamlit. This application analyzes news headlines or news content and predicts whether the news is **Real** or **Fake**.

---

## 🚀 Project Overview

The spread of misinformation through online platforms has become a major challenge. This project uses Machine Learning techniques to automatically classify news articles as genuine or fake based on textual content.

The system is trained on a large dataset of real and fake news articles and provides predictions through an interactive Streamlit web application.

---

## ✨ Features

- Detects whether a news article is **Real** or **Fake**
- User-friendly Streamlit web interface
- NLP-based text preprocessing
- TF-IDF Vectorization for feature extraction
- Logistic Regression Machine Learning model
- Real-time prediction system
- Model persistence using Joblib

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Libraries & Frameworks
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Streamlit
- Joblib

### Machine Learning
- Logistic Regression
- TF-IDF Vectorization

---

## 📂 Project Structure

```text
fake_news_detector/
│
├── dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── model/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset Information

The project uses two publicly available datasets:

### Fake News Dataset
- Contains fake and misleading news articles

### True News Dataset
- Contains verified and genuine news articles

Total dataset size:
- Fake News Articles: ~23,000+
- Real News Articles: ~21,000+
- Total Records: ~44,000+

---

## ⚙️ How It Works

### 1️⃣ Data Collection
The system uses datasets containing both fake and real news articles.

### 2️⃣ Data Preprocessing
- Convert text to lowercase
- Remove punctuation
- Remove stopwords
- Clean and normalize text

### 3️⃣ Feature Extraction
TF-IDF Vectorization converts text into numerical features.

### 4️⃣ Model Training
Logistic Regression is trained on the processed dataset.

### 5️⃣ Prediction
The trained model predicts whether the given news is Real or Fake.

---

## 🎯 Model Performance

| Metric | Value |
|----------|----------|
| Accuracy | 94.14% |
| Algorithm | Logistic Regression |
| Feature Extraction | TF-IDF |

---

## ▶️ Installation

### Clone Repository

```bash
git clone https://github.com/srijabojja-07/fake-news-detector.git
```

### Navigate to Project Folder

```bash
cd fake-news-detector
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will launch in your browser.

---

## 📸 Application Preview

- Enter a news headline or article
- Click **Check News**
- Receive prediction as:
  - ✅ Real News
  - ❌ Fake News

---

## 🔮 Future Enhancements

- Live News API Integration
- Fake News Probability Score
- Deep Learning Models (LSTM/BERT)
- News Source Credibility Analysis
- Multi-language Support
- Real-Time News Verification

---

## 👩‍💻 Author

**Srija Bojja**

B.Tech Information Technology

Aspiring Data Analyst | Cybersecurity Enthusiast | AI & ML Learner

GitHub:
https://github.com/srijabojja-07

---

## ⭐ If you found this project useful, consider giving it a star!
