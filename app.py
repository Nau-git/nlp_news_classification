import streamlit as st
import tensorflow as tf
import pandas as pd


st.title("News Classification")

@st.cache(allow_output_mutation=True)
def teachable_machine_classification(text, weights_file):
    # Load the model
    #model = tf.saved_model.load(weights_file)
    model = tf.keras.models.load_model(weights_file)


    # run the inference
    predictions = model.predict([text])
    predictions = predictions.argmax(axis=1)

    return  predictions

st.header("Please input news title and description to be classified:")
userTitle = st.text_input('Input title here')
userTitle = pd.DataFrame([userTitle], columns=['clean_title'])
userDesc = st.text_area('Input description here')
userDesc = pd.DataFrame([userDesc], columns=['clean_desc'])
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


