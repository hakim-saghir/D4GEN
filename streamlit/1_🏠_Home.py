import streamlit as st

st.title("FA Probability Predictor")

#Buttons
st.markdown("## Welcome to the AIAF-Stroke app !")

st.image("https://i.pinimg.com/originals/93/34/36/933436fb8fb028fe443ac612a88c32e8.jpg")
st.markdown("### Navigate throught the app using the sidebar to access the different pages.")

st.write("* Form: Enter the patient's data and get the FA probability")
st.write("* Import CSV: Import a CSV file containing the patient's data and get the FA probability")
st.write("* Retraining: Retrain the model with new data")
