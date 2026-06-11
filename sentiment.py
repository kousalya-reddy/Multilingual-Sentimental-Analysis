import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("multilingual_sentiment_dataset.csv")

# Features and labels
X = data['text']
y = data['sentiment']

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))

# User input
text = input("Enter a review: ")

text_vector = vectorizer.transform([text])
prediction = model.predict(text_vector)

print("Sentiment:", prediction[0])