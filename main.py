import streamlit as st
import string
import pickle
import nltk
import sklearn
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer


def transform_message(message):
    message = message.lower()
    message = nltk.word_tokenize(message)

    # Remove the alpha Numeric values from messages
    y = []
    for i in message:
        if i.isalnum():
            y.append(i)

    # Remove the stopwords and punctuations
    message = y[:]  # Cloning the y list
    y.clear()
    for i in message:
        if i not in stopwords.words('english') and i not in string.punctuation:  # Removes the punctuations
            # and stopwords from text
            y.append(i)

    # Stemming

    message = y[:]
    y.clear()
    for i in message:
        y.append(ps.stem(i))

    return " ".join(y)


vectorizer = pickle.load(open('Vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.header("Email Classifier")

input_mail = st.text_input("Enter your message to check: ")

if st.button("Predict"):

    # transform
    # transformed_mail = transform_message(input_mail)

    # Vectorize
    vectorized_mail = vectorizer.transform([input_mail])

    # Predict
    result = model.predict(vectorized_mail)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
