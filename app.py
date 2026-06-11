import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load multilingual dataset
data = pd.read_csv("multilingual_sentiment_dataset.csv")

# Features and labels
X = data['text']
y = data['sentiment']

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X, y)

# Streamlit UI
st.title("Multilingual Sentiment Analysis")

text = st.text_input("Enter your review")

if st.button("Predict Sentiment"):
    
    if text != "":
        text_vector = vectorizer.transform([text])
        prediction = model.predict(text_vector)

        st.success("Sentiment: " + prediction[0])
    else:
        st.warning("Please enter a review")