import tensorflow
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pickle
import streamlit as st
# Initialize NLTK resources
nltk.download('stopwords')
ps = PorterStemmer()
def prediction(text):
    corpus = []
    for i in range(0, len(text)):
        review = re.sub(r'[^a-zA-Z0-9]', ' ', text[i])  # Remove special characters
        review = review.lower()  # Lowercase
        review = review.split()  # Split into words
        review = [ps.stem(word) for word in review if word not in stopwords.words('english')]  # Stem and remove stopwords
        review = ' '.join(review)  # Join back into a string
        corpus.append(review)  # Add to corpus
        # Apply one-hot encoding to the entire text
        onehot_rep = [one_hot(" ".join(words),5000) for words in corpus]
        # Pad the sequences to ensure the same length
        pad= pad_sequences(onehot_rep, padding='pre', maxlen=30)
        return model.predict(pad)[0]
# Sample text
st.header('EMAIL CLASSIFIER')
text = st.text_area('ENTER YOUR MAIL/SMS')
text=[text]
# Load the pre-trained model
model = pickle.load(open('model1.pkl', 'rb'))

if st.button('Predict'):
    result = prediction(text)
    if(result==1):
        st.success("SPAM")
    if(result==0):
        st.success("Not SPAM")

