import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import lime
import lime.lime_text

# Load dataset (placeholder path, replace with actual Kaggle dataset path)
data = pd.read_csv("data/fake_or_real_news.csv")  # columns: 'text', 'label'

X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Explain with LIME
class_names = ['REAL', 'FAKE']
explainer = lime.lime_text.LimeTextExplainer(class_names=class_names)

idx = 5
sample_text = X_test.iloc[idx]
print("Sample:", sample_text)
print("True label:", y_test.iloc[idx])

exp = explainer.explain_instance(sample_text, model.predict_proba, num_features=6, labels=[0,1])
exp.save_to_file("lime_explanation.html")
