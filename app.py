import streamlit as st
import numpy as np
import pickle
from sklearn.datasets import load_wine

with open('scaler.pkl',"rb") as file:
    scaler = pickle.load(file)
with open('wine_classifier.pkl',"rb") as file:
    model = pickle.load(file)

wine_data = load_wine()
feature_names = wine_data.feature_names
target_names = wine_data.target_names

# Streamlit app
st.title("Wine Classification App")
st.write("Provide the feature values to classify the type of wine.")

# Create input fields for each feature
user_inputs = []
for feature in feature_names:
    value = st.number_input(f"Enter value for {feature}", step=0.01)
    user_inputs.append(value)
    
if st.button("Predict"):
    input_array = np.array(user_inputs).reshape(1,len(feature_names)) #2d numpy array 
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)
    predicted_class = target_names[prediction[0]]
    st.success(f"Predicted Wine Class is: {predicted_class}")
