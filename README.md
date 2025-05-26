# IA Type Classifier – Machine Learning Project

This project implements a simple text classifier that predicts the **type of artificial intelligence** being described in a given use case.

The model is trained using a small, custom dataset with labeled sentences and uses `scikit-learn`'s `MultinomialNB` algorithm to classify the input.

## 🧠 What it predicts:
The classifier can recognize and label the following types of AI:

- Narrow AI (IA débil)
- General AI (IA general)
- Supervised learning (Aprendizaje supervisado)
- Unsupervised learning (Aprendizaje no supervisado)
- Reinforcement learning (Aprendizaje por refuerzo)

## 🧪 How it works:
- The input is a short description of an AI use case (e.g. "A robot learns by trial and error").
- The sentence is vectorized using `CountVectorizer`.
- The trained model predicts the most likely AI type.

## 💻 Technologies:
- Python
- Scikit-learn
- CountVectorizer
- Naive Bayes Classifier

## 🚀 How to run:

1. Clone the repository or download the files.
2. Install the required library:
```bash
pip install scikit-learn

