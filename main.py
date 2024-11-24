import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk import download
from textblob import Word, TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

train = pd.read_csv("/content/train.csv")
test = pd.read_csv("/content/test.csv")
submission = pd.read_csv("/content/sample_submission.csv")


train["keyword"].fillna(train["keyword"].mode()[0], inplace=True)
train["location"].fillna(train["location"].mode()[0], inplace=True)

train["text"] = train["text"].str.lower().str.replace("[^\w\s]", '', regex=True).str.replace("\d+", '', regex=True)

download('stopwords')
sw = stopwords.words("english")
train["text"] = train["text"].apply(lambda x: " ".join(word for word in x.split() if word not in sw))

download('wordnet')
train["text"] = train["text"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(train["text"])
y = train["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

new_text = ["An alert rang on everyone's phones, but it was just a test of the emergency system"
]
new_text_transformed = vectorizer.transform(new_text)
prediction = log_model.predict(new_text_transformed)
print(f"Prediction for the new tweet: {prediction}")
