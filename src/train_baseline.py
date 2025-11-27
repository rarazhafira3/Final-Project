# src/train_baseline.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
test = pd.read_csv("data/processed/test.csv")

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=3)),
    ("clf", LogisticRegression(max_iter=1000))
])

pipe.fit(train['text'], train['label'])
pred = pipe.predict(test['text'])

print(classification_report(test['label'], pred))
joblib.dump(pipe, "models/baseline_tfidf_lr.joblib")
