import streamlit as st
import pickle as pkl
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


tfidf=pkl.load(open('vectorizer.pkl','rb'))
mnb=pkl.load(open('model.pkl','rb'))
st.title("Spam Classifier")



ps=PorterStemmer()
ps.stem("loving")

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text :
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text=y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)


input_sms = st.text_area("Enter your msg/email here")
if st.button("Check"):
  

    transformed_sms=transform_text(input_sms)

    vectorized_sms=tfidf.transform([transformed_sms])

    result=mnb.predict(vectorized_sms)[0]
    if result==1:
        st.header("Spam")
    else:
        st.header("Not a spam")





