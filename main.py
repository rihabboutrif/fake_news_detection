import streamlit as st
import pickle
import joblib


from text_cleanin import clean_text
from sklearn.base import BaseEstimator, TransformerMixin




class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        # Assume X is a list of strings
        return [clean_text(text) for text in X]  # Apply cleaning logic to each string


# Title and description
loaded_pipeline = joblib.load('pipeline.pkl')
model = joblib.load('model1.pkl')
vectorizer = joblib.load('vectorizer.pkl')



st.title("Fake News Detection")
st.write("Enter a news headline or text to check if it's fake or real.")

# Input text
user_input = st.text_area("Enter News Text", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.error("Please enter some text.")
    else:
         # Clean the input (as a string)
        cleaned_input = loaded_pipeline.transform([user_input])[0]
        
        # Transform cleaned text into a feature vector
        input_vectorized = vectorizer.transform([cleaned_input])
        
        # Predict the label
        predicted_label = model.predict(input_vectorized)
        print("Prediction:", predicted_label)  # Output the prediction
        # Show results
        if predicted_label == 1:
            st.error(f"🚨 Fake News Detected!")
        else:
            st.success(f"✅ Real News!")
