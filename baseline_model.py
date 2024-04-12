import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

# Data preparation
column_labels = ['label', 'text']
train_df = pd.read_csv('cooperator/data/fulltrain.csv', header=None, names=column_labels)
test__df = pd.read_csv('cooperator/data/balancedtest.csv', header=None, names=column_labels)

data_df = pd.concat([train_df, test__df], ignore_index=True)
data_labels_df = data_df["label"]
data_texts__df = data_df["text"]

X_train, X_test, y_train, y_test = train_test_split(
    data_texts__df,
    data_labels_df,
    test_size=0.2,
    random_state=45,
    stratify=data_labels_df
)

# Vectorise
token_pattern = r'(?u)\b[A-Za-z][A-Za-z]+\b'
tfidf_vectoriser = TfidfVectorizer(token_pattern=token_pattern, stop_words='english', max_df=0.9)
tfidf_train = tfidf_vectoriser.fit_transform(X_train)
tfidf_test = tfidf_vectoriser.transform(X_test)
print(tfidf_vectoriser.get_feature_names_out()[:400])
print(tfidf_train.A[:5])
num_features = len(tfidf_vectoriser.vocabulary_)
print(f'Number of features: {num_features}')
print(f'Shape of TF-IDF matrix: {tfidf_train.shape}')
print(f'Number of features: {tfidf_train.shape[1]}')

# Train model
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(tfidf_train, y_train)
predictions = model.predict(tfidf_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

def get_insights(dataframe):
    print(dataframe.head())
    print(dataframe.shape)
    print(dataframe.info())
    print(dataframe["label"].value_counts())