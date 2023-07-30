import streamlit as st
import tensorflow as tf
import pandas as pd
import re, string, nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

st.title("News Classification")

@st.cache_resource
def teachable_machine_classification(text, weights_file):
    # Load the model
    #model = tf.saved_model.load(weights_file)
    model = tf.keras.models.load_model(weights_file)


    # run the inference
    predictions = model.predict([text])
    predictions = predictions.argmax(axis=1)

    return  predictions

# preprocessing
@st.cache_resource
def preprocess(text):
    text = text.lower() 
    text = text.strip()  
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text) 
    text = re.sub('\s+', ' ', text) 
    text = re.sub(r'[^\w\s]', ' ', str(text).lower().strip()) 
    text = re.sub(r'\d',' ',text) 
    
    return text

def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

wl = WordNetLemmatizer()
 
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) 
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] 
    return " ".join(a)

def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))

st.header("Please input news title and description to be classified:")
userTitle = st.text_input('Input title here')
userTitle = pd.DataFrame([userTitle], columns=['clean_title'])
userDesc = st.text_area('Input description here')
userDesc = pd.DataFrame([userDesc], columns=['clean_desc'])

userTitle['clean_title'] = userTitle['clean_title'].apply(lambda x: finalpreprocess(x))
userDesc['clean_desc'] = userDesc['clean_desc'].apply(lambda x: finalpreprocess(x))

userInput = [userTitle, userDesc]

if st.button('Predict'):
    st.write("Predicting...")
    label = teachable_machine_classification(userInput, 'tuned_model')
    if label == 0:
        st.write("It's a World news article")
    elif label == 1:
        st.write("It's a Sports news article")
    elif label == 2:
        st.write("It's a Business news article")
    elif label == 3:
        st.write("It's a Sci/Tech news article")
    else:
        st.write("Something went wrong")
else:
    st.write('')


