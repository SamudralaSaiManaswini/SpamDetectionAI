# 🧠 AI Project – Spam Message Detection

This project is an AI-based spam detection system that classifies messages as "Spam" or "Not Spam" using Natural Language Processing (NLP) and a Naive Bayes classifier. The code was developed using basic tools — Notepad for writing the Python code and Anaconda Prompt for execution — demonstrating how simple tools can be used to build powerful AI systems.

---

## 📌 Project Objective

The main goal of this project is to detect whether a given message is spam or legitimate (ham). By analyzing a dataset of past SMS messages, the machine learning model learns patterns in spam messages and predicts on new inputs in real time.

---

## 💻 Tools & Technologies Used

- 📄 Notepad – for writing Python scripts
- 🐍 Python 3.x
- ⚙️ Anaconda Prompt – to execute the code
- 📦 Libraries:
  - pandas – data manipulation
  - scikit-learn – ML model (Naive Bayes)
  - nltk – text preprocessing (stopwords)
  - matplotlib & seaborn – evaluation graphs (confusion matrix)

---

## 📁 Dataset

- Name: SMS Spam Collection Dataset
- Source: [Kaggle – UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Format: CSV
- Description: 5,572 labeled messages (ham/spam)

---

## ⚙️ Features & Workflow

1. Load the dataset
2. Clean and preprocess the text (remove stopwords, punctuation)
3. Convert messages into numerical vectors using CountVectorizer
4. Train a Multinomial Naive Bayes classifier
5. Evaluate using accuracy and confusion matrix
6. Accept user input and predict whether the message is spam or not

---

## 🚀 How to Run the Project

1. Clone or download this project folder.
2. Place the dataset file (spam.csv) in the dataset/ folder.
3. Open Anaconda Prompt.
4. Navigate to the project directory.
5. Install required libraries:
6. Run the script:
7. Enter any message when prompted to test the model.

---

## ✅ Sample Test

```text
Input:  Congratulations! You've won a free recharge!
Output: 🔴 This is Spam.

Input:  Hey, are we meeting at 5 PM?
Output: 🟢 This is Not Spam.
