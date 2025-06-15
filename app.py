import streamlit as st
import pandas as pd
import numpy as np
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download stopwords (only once)
nltk.download('stopwords')
from nltk.corpus import stopwords

# Page configuration
st.set_page_config(page_title="Spam Detection AI", layout="centered")
st.title("📧 Spam Message Detector using AI")

# Load dataset function
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/spam.csv", encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label_num'] = df.label.map({'ham': 0, 'spam': 1})
    return df

data = load_data()

# Preprocess text function
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

data['clean_msg'] = data['message'].apply(clean_text)

# Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['clean_msg'])
y = data['label_num']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Sidebar - show accuracy and confusion matrix
st.sidebar.header("📊 Model Stats")
st.sidebar.write(f"✅ Accuracy: {round(accuracy*100, 2)}%")

if st.sidebar.checkbox("Show Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.sidebar.pyplot(fig)

# Input & Prediction
st.subheader("Try It Yourself 👇")
user_input = st.text_area("Enter your message here:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        cleaned_input = clean_text(user_input)
        input_vector = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vector)[0]

        if prediction == 1:
            st.error("🔴 This is Spam.")
        else:
            st.success("🟢 This is Not Spam.")
