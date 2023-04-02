import streamlit as st
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import xgboost as xgb

######################################### Functions ###################################################
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


def loadAndPredict(model_name, patient_index):
    # Load the JSON file
    with open('./D4GEN/streamlit/data/models/model.json', 'r') as file:
        model_json = file.read()

    with open('./D4GEN/streamlit/data/patient_database/{}.csv'.format(patient_index), 'r') as file:
        test_data = file.read()
    # Load the JSON model into an XGBoost Booster
    booster = xgb.Booster()
    st.markdown(model_json)
    booster.load_model('./D4GEN/streamlit/data/models/model.json')

    # Prepare your test data as a DMatrix object
    test_data = xgb.DMatrix('./D4GEN/streamlit/data/patient_database/{}.csv'.format(patient_index))

    # Make predictions using the booster object
    predictions = booster.predict(test_data)

    save_prediction(patient_index, predictions)

    return predictions

def save_prediction(patient_index, prediction):
    # Create a dataframe with the prediction
    df = pd.DataFrame(prediction, columns=['prediction'])

    # Save the dataframe as a CSV file
    df.to_csv('./D4GEN/streamlit/data/metrics/{}.csv'.format(patient_index), index=False)


############################################## Streamlit ####################################################################




# Set title of the page
st.title("Inference on Form")

# Input field for patient index
patient_index = st.number_input("Patient Index", step=1)

# Create two columns for input fields
col1, col2 = st.columns(2)

# Input fields for various parameters
with col1:
    AGE = st.number_input("Age", min_value=0, max_value=100, step=1)
    ATCD_CONSTATE = st.selectbox(
        "ATCD constaté", ["Jamais", "Occasionnelle", "Régulière"])
    ICI_IRM_LAC = st.selectbox("Ici irm lac", ["Oui", "Non"])
    ICI_ASPECT = st.number_input(
        "Ici aspect", min_value=0, max_value=10, step=1)
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
    BIO_FERRITINE = st.number_input(
        "BIO FERRITINE", min_value=0.0, max_value=1000.0, step=0.1)
    ATCD_CONSO_ALCOOL = st.selectbox(
        "ATCD conso alcool", ["Jamais", "Occasionnelle", "Régulière"])
    ATCD_HTA = st.selectbox("ATCD HTA", ["Oui", "Non"])
    HISTO_DEFICIT_MOTEUR = st.selectbox(
        "Histo déficit moteur", ["Oui", "Non"])
    ICI_FLAIR_SEQAVC = st.selectbox("Ici flair seqavc", ["Oui", "Non"])
    THROMBECTOMIE_MECANIQUE = st.selectbox(
        "Thrombectomie mécanique", ["Oui", "Non"])
    ICI_IRM_NONLAC_D_ACP = st.selectbox("Ici irm nonlac d acp", ["Oui", "Non"])
    ICI_IRM_NONLAC_D_IPP = st.selectbox("Ici irm nonlac d ipp", ["Oui", "Non"])
    ICI_IRM_NONLAC_D_AITC = st.selectbox("Ici irm nonlac d aitc", ["Oui", "Non"])
    ICI_IRM_NONLAC_D_ACPI = st.selectbox("Ici irm nonlac d acpi", ["Oui", "Non"])
    ICI_IRM_NONLAC_D_ACS = st.selectbox("Ici irm nonlac d acs", ["Oui", "Non"])
    ICI_IRM_NONLAC_G_ACP = st.selectbox("Ici irm nonlac g acp", ["Oui", "Non"])
    ICI_IRM_NONLAC_G_IPP = st.selectbox("Ici irm nonlac g ipp", ["Oui", "Non"])
    ICI_IRM_NONLAC_G_AITC = st.selectbox("Ici irm nonlac g aitc", ["Oui", "Non"])
    ICI_IRM_NONLAC_G_ACAI = st.selectbox("Ici irm nonlac g acai", ["Oui", "Non"])
    ECPL_BIO_LDL = st.number_input(
        "ECPL BIO LDL", min_value=0.0, max_value=5.0, step=0.1)
    
    

with col2:
    SEXE = st.selectbox("Sexe", ["M", "F"])
    UNITE_ALCOOL_SEM = st.number_input(
        "Unité d'Alcool par Semaine", min_value=0, max_value=300, step=1)
    NIHSS_INITIAL = st.number_input(
        "NIHSS Initial", min_value=0, max_value=32, step=1)
    ICI_IRM_NONLAC_D_ACM = st.selectbox("Ici irm nonlac d acm", ["Oui", "Non"])
    ICI_SWAN_THROMBUS = st.selectbox("Ici swan thrombus", ["Oui", "Non"])
    ETIO_TOAST = st.selectbox("ETIO TOAST", ["Athérothrombotique", "Cardioembolique",
                            "Lacunaire", "Autre cause", "Indéterminé A", "Indéterminé B", "Indéterminé C"])
    BIO_HBA1C = st.number_input(
        "BIO HBA1C", min_value=3.0, max_value=10.0, step=0.1)
    BIO_CPK = st.number_input("BIO CPK", min_value=0, max_value=10000, step=1)
    BIO_TSH3G = st.number_input(
        "BIO TSH3G", min_value=0.000, max_value=10.000, step=0.001)
    ATCD_DIABETE = st.selectbox("ATCD diabète", ["Oui", "Non"])
    INTUITION_MEDICALE_FA = st.selectbox(
        "Intuition médicale FA", ["Oui", "Non"])
    HISTO_APHASIE = st.selectbox("Histo aphasie", ["Oui", "Non"])
    THROMBOLYSE_IV = st.selectbox("Thrombolyse IV", ["Oui", "Non"])
    ICI_IRM_NONLAC_D_ACA = st.selectbox("Ici irm nonlac d aca", ["Oui", "Non"])
    ICI_IRM_NONLAC_D_ACHA = st.selectbox("Ici irm nonlac d acha", ["Oui", "Non"])
    ICI_IRM_NONLAC_D_LB = st.selectbox("Ici irm nonlac d lb", ["Oui", "Non"])
    ICI_IRM_NONLAC_G_ACM = st.selectbox("Ici irm nonlac g acm", ["Oui", "Non"])
    ICI_IRM_NONLAC_D_ACAI = st.selectbox("Ici irm nonlac d acai", ["Oui", "Non"])
    ICI_IRM_NONLAC_G_ACA = st.selectbox("Ici irm nonlac g aca", ["Oui", "Non"])
    ICI_IRM_NONLAC_G_ACHA = st.selectbox("Ici irm nonlac g acha", ["Oui", "Non"])
    ICI_IRM_NONLAC_G_LB = st.selectbox("Ici irm nonlac g lb", ["Oui", "Non"])
    ICI_IRM_NONLAC_G_ACPI = st.selectbox("Ici irm nonlac g acpi", ["Oui", "Non"])
    ICI_IRM_NONLAC_G_ACS = st.selectbox("Ici irm nonlac g acs", ["Oui", "Non"])
    ECPL_BIO_HBA1C = st.selectbox("ECPL BIO HBA1C", ["Oui", "Non"])
    OG_ETAT = st.selectbox("OG ETAT", ["Oui", "Non"])



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
            "ICI_IRM_LAC": ICI_IRM_LAC,
            "ICI_IRM_NONLAC_D_ACM": ICI_IRM_NONLAC_D_ACM,
            "ICI_ASPECT": ICI_ASPECT,
            "ICI_SWAN_THROMBUS": ICI_SWAN_THROMBUS,
            "ETIO_TOAST": ETIO_TOAST,
            "BIO_POTASSIUM": BIO_POTASSIUM,
            "BIO_CPK": BIO_CPK,
            "BIO_TROPONINE": BIO_TROPONINE,
            "BIO_CRP": BIO_CRP,
            "BIO_TSH3G": BIO_TSH3G,
            "ATCD_CONSO_ALCOOL": ATCD_CONSO_ALCOOL,
            "ATCD_DIABETE": ATCD_DIABETE,
            "ATCD_HTA": ATCD_HTA,
            "INTUITION_MEDICALE_FA": INTUITION_MEDICALE_FA,
            "HISTO_DEFICIT_MOTEUR": HISTO_DEFICIT_MOTEUR,
            "HISTO_APHASIE": HISTO_APHASIE,
            "ICI_FLAIR_SEQAVC": ICI_FLAIR_SEQAVC,
            "THROMBOLYSE_IV": THROMBOLYSE_IV,
            "THROMBECTOMIE_MECANIQUE": THROMBECTOMIE_MECANIQUE,
            "ICI_IRM_NONLAC_D_ACA": ICI_IRM_NONLAC_D_ACA,
            "ICI_IRM_NONLAC_D_ACP": ICI_IRM_NONLAC_D_ACP,
            "ICI_IRM_NONLAC_D_ACHA": ICI_IRM_NONLAC_D_ACHA,
            "ICI_IRM_NONLAC_D_IPP": ICI_IRM_NONLAC_D_IPP,
            "ICI_IRM_NONLAC_D_LB": ICI_IRM_NONLAC_D_LB,
            "ICI_IRM_NONLAC_D_AITC": ICI_IRM_NONLAC_D_AITC,
            "ICI_IRM_NONLAC_D_ACPI": ICI_IRM_NONLAC_D_ACPI,
            "ICI_IRM_NONLAC_D_ACAI": ICI_IRM_NONLAC_D_ACAI,
            "ICI_IRM_NONLAC_D_ACS": ICI_IRM_NONLAC_D_ACS,
            "ICI_IRM_NONLAC_G_ACM": ICI_IRM_NONLAC_G_ACM,
            "ICI_IRM_NONLAC_G_ACA": ICI_IRM_NONLAC_G_ACA,
            "ICI_IRM_NONLAC_G_ACP": ICI_IRM_NONLAC_G_ACP,
            "ICI_IRM_NONLAC_G_ACHA": ICI_IRM_NONLAC_G_ACHA,
            "ICI_IRM_NONLAC_G_IPP": ICI_IRM_NONLAC_G_IPP,
            "ICI_IRM_NONLAC_G_LB": ICI_IRM_NONLAC_G_LB,
            "ICI_IRM_NONLAC_G_AITC": ICI_IRM_NONLAC_G_AITC,
            "ICI_IRM_NONLAC_G_ACPI": ICI_IRM_NONLAC_G_ACPI,
            "ICI_IRM_NONLAC_G_ACAI": ICI_IRM_NONLAC_G_ACAI,
            "ICI_IRM_NONLAC_G_ACS": ICI_IRM_NONLAC_G_ACS,
            "ECPL_BIO_NTPROBNP": ECPL_BIO_NTPROBNP,
            "NIHSS_INITIAL": NIHSS_INITIAL,
            "UNITE_ALCOOL/SEM": UNITE_ALCOOL_SEM,
            "ECPL_BIO_LDL": ECPL_BIO_LDL,
            "OG_ETAT": OG_ETAT,
            "ECPL_BIO_HBA1C": ECPL_BIO_HBA1C

            }

        return pd.DataFrame(patient_data, index=[0])

    # Load data and display the shape of the dataframe
    df_patient_data = load_data()
    t_df = df_patient_data.T


    # Progress bar to show operation in progress
    progress_text = "Operation in progress. Please wait."

    my_bar = st.progress(0, text=progress_text)

    # Loop to update the progress bar
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    path = r"./D4GEN/streamlit/data/patient_database/{}.csv".format(patient_index)
    df_patient_data.to_csv(path, index=False)
    st.success("New dataset saved !", icon="✅")
    

    # Display the prediction
    with st.expander("Results"):

        # Load the prediction
        
        #prediction = loadAndPredict("model", patient_index)

        df = pd.read_csv("./D4GEN/streamlit/data/patient_database/{}.csv".format(patient_index))
        st.markdown("## Results")
        #st.write("The patient has a {}% chance of having a good outcome.".format(prediction))
        st.dataframe(df.T, use_container_width=True)
        st.cache_data.clear()

    # Display the prediction
with st.expander("Model Info"):

    tab1, tab2 = st.tabs(["Model Info", "Model Parameters"])

    with tab1:

        st.image("https://media.giphy.com/media/3o7TKSjRrfIPjeiVyQ/giphy.gif", use_column_width=True)
   
    with tab2:

        st.markdown("Model metrics")

        arr = np.random.normal(1, 1, size=100)
        fig, ax = plt.subplots()
        ax.hist(arr, bins=20)
        st.pyplot(fig)

    
