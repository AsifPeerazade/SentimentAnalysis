import streamlit as st
import os
import matplotlib.pyplot as plt
import numpy as np
import nltk
import re
import string
import pickle
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()
import nltk
nltk.download('stopwords')
stopwords_list = nltk.corpus.stopwords.words('english')
stopwords_list.remove('no')
stopwords_list.remove('not')
model = pickle.load(open('SentimentAnalysis.p','rb'))

st.title("Sentiment Analysis")
st.subheader("Enter Text: ")
text = st.text_input(" ")
def remove_sp(text):
        text = text.lower()
        text = re.sub('\[.*?\]',"",text)
        text = re.sub('[%s]' %re.escape(string.punctuation), "", text)
        text = re.sub('\w*\d\w',"",text)
        text = re.sub('[''""_]', "", text)
        text = re.sub('\n',"", text)
        return text
    
    #remove stopwords
def remove_stopwords(text):
  tokens = tokenizer.tokenize(text)
  tokens = [token.strip() for token in tokens]
  filtered_tokens = [token for token in tokens if token not in stopwords_list]
  filtered_text = ' '.join(filtered_tokens)
  return filtered_text

text = text.lower()
text = remove_sp(text)
text = remove_stopwords(text)
st.write(text)
text = [text]
y_out = model.predict(text)

if st.button("Predict"):
    
    st.write(f'PREDICTED OUTPUT: {y_out}')

    if (y_out == "Positive"):
        st.write("\U0001f600")
    else:
        st.write("😞")
    
    CATEGORIES = ['Negative', 'Positive']
    q = model.predict_proba(text)
    for index, item in enumerate(CATEGORIES):
          st.write(f'{item} : {q[0][index]*100}%')
    
