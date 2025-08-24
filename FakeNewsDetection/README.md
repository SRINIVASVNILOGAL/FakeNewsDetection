# Fake News & AI-Generated News Detection with Explainability

## Overview
This project builds an NLP-based classifier to detect Fake vs Real news articles. 
It also extends to detect AI-generated vs Human-written news. 
The project uses TF-IDF + Logistic Regression for classification and LIME for explainability.

## Features
- Text preprocessing (tokenization, stopword removal, lemmatization)
- Feature extraction (TF-IDF, embeddings)
- Classification using Logistic Regression
- Explainability using LIME
- Extension: Detect AI vs Human-written text

## Dataset
- Fake News dataset: [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data)
- AI-generated dataset: Mix of human news (BBC/Reuters) and AI-generated text (GPT-based)

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Run `fake_news_detection.py`

## Learning Outcomes
- NLP pipeline (preprocessing → feature extraction → classification)
- TF-IDF math and logistic regression math
- Explainable AI using LIME
- Extending ML to GenAI detection
