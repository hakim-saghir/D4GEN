# Import necessary libraries
import streamlit as st
import pandas as pd
import time
import os
import tensorflow as tf

# Set page title
st.set_page_config(page_title="Form")

# Function to perform computation on the data
def perform_computation(data):
    result = "Health AI Result: " + str(data)
    return result

# Function to normalize column names
def normalize_column(column_name: str):
        return str(column_name.replace(' ', '_') \
                            .replace('é', 'e') \
                            .replace('à', 'a') \
                            .replace("'", '_').upper())

def predict(data):

    model = tf.keras.models.load_model('')
    prediction = model.predict(data)
    return prediction



###########################################################################################################################
#
#
#
###########################################################################################################################

# Set title of the page
st.title("Inference on Form")

# Input field for patient index
patient_index = st.number_input("Patient Index", step=1)

# Create two columns for input fields
col1, col2 = st.columns(2)

# Input fields for various parameters
with col1:
    AGE = st.number_input("Age", min_value=0, max_value=120, step=1)
    ATCD_CONSTATE = st.selectbox(
        "ATCD constaté", ["Jamais", "Occasionnelle", "Régulière"])
    ATCD_FA_CONNU = st.selectbox("ATCD FA connu", ["Oui", "Non"])
    ICI_IRM_LAC = st.selectbox("Ici irm lac", ["Oui", "Non"])
    ICI_ASPECT = st.number_input(
        "Ici aspect", min_value=0, max_value=10, step=1)
    ECG_A_ARRIVEE = st.selectbox(
        "ECG à l'arrivée", ["Fibrillation", "Sinusal"])
    ECPL_ETT = st.selectbox("ECPL ETT", ["Oui", "Non"])
    ECPL_BIO_NTPROBNP = st.number_input(
        "ECPL BIO NTPROBNP", min_value=0, max_value=10000, step=1)
    BIO_POTASSIUM = st.number_input(
        "BIO POTASSIUM", min_value=2.0, max_value=10.0, step=0.1)
    BIO_LDL = st.number_input(
        "BIO LDL", min_value=0.0, max_value=5.0, step=0.1)
    BIO_TROPONINE = st.number_input(
        "BIO TROPONINE", min_value=0.00, max_value=100.00, step=0.01)
    BIO_CRP = st.number_input("BIO CRP", min_value=0.0,
                            max_value=100.0, step=0.1)

with col2:
    SEXE = st.selectbox("Sexe", ["M", "F"])
    UNITE_ALCOOL_SEM = st.number_input(
        "Unité d'Alcool par Semaine", min_value=0, max_value=300, step=1)
    NIHSS_INITIAL = st.number_input(
        "NIHSS Initial", min_value=0, max_value=32, step=1)
    ICI_IRM_NONLAC_D_ACM = st.selectbox("Ici irm nonlac d acm", ["Oui", "Non"])
    ICI_SWAN_THROMBUS = st.selectbox("Ici swan thrombus", ["Oui", "Non"])
    FA_SUR_ECG = st.selectbox("FA sur ECG", ["1", "0"])
    ECPL_ETT_TXT = st.text_input("ECPL ETT TXT")
    ETIO_TOAST = st.selectbox("ETIO TOAST", ["Athérothrombotique", "Cardioembolique",
                            "Lacunaire", "Autre cause", "Indéterminé A", "Indéterminé B", "Indéterminé C"])
    BIO_HBA1C = st.number_input(
        "BIO HBA1C", min_value=3.0, max_value=10.0, step=0.1)
    BIO_CPK = st.number_input("BIO CPK", min_value=0, max_value=10000, step=1)
    ECPL_BIO_NTPROBNP = st.number_input(
        "ECPL BIO NTPROBNP", min_value=0, max_value=50000, step=1)
    BIO_TSH3G = st.number_input(
        "BIO TSH3G", min_value=0.000, max_value=10.000, step=0.001)



# Button to save the input data and perform prediction
if st.button("Save & Predict"):

    # When the button is clicked, set 'show_element' to False
    show_element = False
    st.session_state.show_element = False

    # Display the element only if 'show_element' is True
    if show_element:
        st.write("This element will be removed when the button is clicked.")

    @st.cache_data # Function to load and cache data
    def load_data():
        # Create a dictionary with input data
        patient_data = {
            "AGE": AGE,
            "SEXE": SEXE,
            "ATCD_CONSTATE": ATCD_CONSTATE,
            "UNITE_ALCOOL_SEM": UNITE_ALCOOL_SEM,
            "ATCD_FA_CONNU": ATCD_FA_CONNU,
            "NIHSS_INITIAL": NIHSS_INITIAL,
            "ICI_IRM_LAC": ICI_IRM_LAC,
            "ICI_IRM_NONLAC_D_ACM": ICI_IRM_NONLAC_D_ACM,
            "ICI_ASPECT": ICI_ASPECT,
            "ICI_SWAN_THROMBUS": ICI_SWAN_THROMBUS,
            "ECG_A_ARRIVEE": ECG_A_ARRIVEE,
            "FA_SUR_ECG": FA_SUR_ECG,
            "ECPL_ETT": ECPL_ETT,
            "ECPL_ETT_TXT": ECPL_ETT_TXT,
            "ECPL_BIO_NTPROBNP": ECPL_BIO_NTPROBNP,
            "ETIO_TOAST": ETIO_TOAST,
            "BIO_POTASSIUM": BIO_POTASSIUM,
            "BIO_HBA1C": BIO_HBA1C,
            "BIO_LDL": BIO_LDL,
            "BIO_CPK": BIO_CPK,
            "BIO_TROPONINE": BIO_TROPONINE,
            "BIO_CRP": BIO_CRP,
            "BIO_TSH3G": BIO_TSH3G}

        return pd.DataFrame(patient_data, index=[0])

    # Load data and display the shape of the dataframe
    df_patient_data = load_data()
    st.write(df_patient_data.shape)
    t_df = df_patient_data.T


    # Progress bar to show operation in progress
    progress_text = "Operation in progress. Please wait."

    my_bar = st.progress(0, text=progress_text)

    # Loop to update the progress bar
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    path = r"./data/patient_database/{}.csv".format(patient_index)
    df_patient_data.to_csv(path, index=False)
    st.success("New dataset saved !", icon="✅")
    st.cache_data.clear()

    # Display the prediction
with st.expander("Results"):
    df = pd.read_csv("data/patient_database/{}.csv".format(patient_index))
    st.markdown("## Results")
    st.dataframe(df.T, use_container_width=True)

    # Display the prediction
with st.expander("Model Info"):
    tab1, tab2 = st.tabs(["Model Info", "Model Parameters"])
    with tab1:
        st.image("https://media.giphy.com/media/3o7TKSjRrfIPjeiVyQ/giphy.gif", use_column_width=True)
    with tab2:
        st.subheader("Model Parameters")
    
