import streamlit as st
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import xgboost as xgb



######################################### Functions ###################################################

# Function to normalize column names
def normalize_column(column_name: str):
    return str(column_name.replace(' ', '_') \
                        .replace('é', 'e') \
                        .replace('à', 'a') \
                        .replace("'", '_').upper())

# Function to preprocess the data
def data_preprocessing(csv_file):

    prepared_dataset =  csv_file.copy()

    # Drop rows and columns with all missing values
    prepared_dataset = prepared_dataset.dropna(how='all')

    # Normalize column names
    for i, c in enumerate(prepared_dataset.columns):
        prepared_dataset = prepared_dataset.rename(columns={c: normalize_column(c)})

    # Select relevant columns
    ml_dataset = prepared_dataset[
        ['AGE', 'ATCD_CONSO_ALCOOL', 'ATCD_DIABETE', 'ATCD_HTA', 'BIO_CPK',
       'BIO_CRP', 'BIO_POTASSIUM', 'ECPL_BIO_HBA1C', 'BIO_TROPONINE',
       'BIO_TSH3G', 'ECPL_BIO_NTPROBNP', 'ETIO_TOAST', 'ICI_ASPECT',
       'ICI_SWAN_THROMBUS', 'NIHSS_INITIAL', 'ICI_IRM_LAC', 'SEXE',
       'UNITE_ALCOOL/SEM', 'HISTO_DEFICIT_MOTEUR', 'HISTO_APHASIE',
       'ICI_FLAIR_SEQAVC', 'THROMBOLYSE_IV', 'THROMBECTOMIE_MECANIQUE',
       'ECPL_BIO_LDL', 'INTUITION_MEDICALE_FA', 'ICI_IRM_NONLAC_D_ACM',
       'ICI_IRM_NONLAC_D_ACA', 'ICI_IRM_NONLAC_D_ACP', 'ICI_IRM_NONLAC_D_ACHA',
       'ICI_IRM_NONLAC_D_IPP', 'ICI_IRM_NONLAC_D_LB', 'ICI_IRM_NONLAC_D_AITC',
       'ICI_IRM_NONLAC_D_ACPI', 'ICI_IRM_NONLAC_D_ACAI',
       'ICI_IRM_NONLAC_D_ACS', 'ICI_IRM_NONLAC_G_ACM', 'ICI_IRM_NONLAC_G_ACA',
       'ICI_IRM_NONLAC_G_ACP', 'ICI_IRM_NONLAC_G_ACHA', 'ICI_IRM_NONLAC_G_IPP',
       'ICI_IRM_NONLAC_G_LB', 'ICI_IRM_NONLAC_G_AITC', 'ICI_IRM_NONLAC_G_ACPI',
       'ICI_IRM_NONLAC_G_ACAI', 'ICI_IRM_NONLAC_G_ACS', 'OG_ETAT']
    ]

    return ml_dataset

# Function to check if the uploaded file has the correct format
def check_data(csv_file):
    try:
        df = pd.read_csv(csv_file)  # Import the csv file

        if df.shape[1] != 46:
            st.error("The number of columns is not correct")
            return False
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty or contains only whitespace.")
        return False
    return True

########################################## Streamlit ################################################################



# Set up the app's title

st.title("Inference on File")

local_time = time.localtime()
local_time = '{}-{}-{}-{}-{}-{}'.format(local_time.tm_year, local_time.tm_mon, local_time.tm_mday, 
                                        local_time.tm_hour, local_time.tm_min, local_time.tm_sec)

# Allow users to upload a CSV file
uploaded_file = st.file_uploader("Upload CSV", type="csv")



# If the uploaded file is not None and passes the check_data() function
if uploaded_file is not None and check_data(uploaded_file):

    # Get the patient's name from the file name
    patient_index = uploaded_file.name.split('.')[0]

    # Display patient's name
    st.subheader("Patient index: " + patient_index)

    # Read the uploaded CSV file
    uploaded_file.seek(0)
    df_patient_data = pd.read_csv(uploaded_file)

    # Preprocess the data
    ml_dataset = data_preprocessing(df_patient_data)

    # If the user clicks on the "Compute" button
    with uploaded_file:
        if st.button("Compute"):

            # Set up two columns for display
            col1, col2= st.columns(2)

            # Cache the data for faster loading
            @st.cache_data
            def load_data():
                patient_data = {
                    "AGE": ml_dataset['AGE'],
                    "ATCD_CONSO_ALCOOL": ml_dataset['ATCD_CONSO_ALCOOL'],
                    "ATCD_DIABETE": ml_dataset['ATCD_DIABETE'],
                    "ATCD_HTA": ml_dataset['ATCD_HTA'],
                    "BIO_CPK": ml_dataset['BIO_CPK'],
                    "BIO_CRP": ml_dataset['BIO_CRP'],
                    "BIO_POTASSIUM": ml_dataset['BIO_POTASSIUM'],
                    "ECPL_BIO_HBA1C": ml_dataset['ECPL_BIO_HBA1C'],
                    "BIO_TROPONINE": ml_dataset['BIO_TROPONINE'],
                    "BIO_TSH3G": ml_dataset['BIO_TSH3G'],
                    "ECPL_BIO_NTPROBNP": ml_dataset['ECPL_BIO_NTPROBNP'],
                    "ETIO_TOAST": ml_dataset['ETIO_TOAST'],
                    "ICI_ASPECT": ml_dataset['ICI_ASPECT'],
                    "ICI_SWAN_THROMBUS": ml_dataset['ICI_SWAN_THROMBUS'],
                    "NIHSS_INITIAL": ml_dataset['NIHSS_INITIAL'],
                    "ICI_IRM_LAC": ml_dataset['ICI_IRM_LAC'],
                    "SEXE": ml_dataset['SEXE'],
                    "UNITE_ALCOOL/SEM": ml_dataset['UNITE_ALCOOL/SEM'],
                    "HISTO_DEFICIT_MOTEUR": ml_dataset['HISTO_DEFICIT_MOTEUR'],
                    "HISTO_APHASIE": ml_dataset['HISTO_APHASIE'],
                    "ICI_FLAIR_SEQAVC": ml_dataset['ICI_FLAIR_SEQAVC'],
                    "THROMBOLYSE_IV": ml_dataset['THROMBOLYSE_IV'],
                    "THROMBECTOMIE_MECANIQUE": ml_dataset['THROMBECTOMIE_MECANIQUE'],
                    "ECPL_BIO_LDL": ml_dataset['ECPL_BIO_LDL'],
                    "INTUITION_MEDICALE_FA": ml_dataset['INTUITION_MEDICALE_FA'],
                    "ICI_IRM_NONLAC_D_ACM": ml_dataset['ICI_IRM_NONLAC_D_ACM'],
                    "ICI_IRM_NONLAC_D_ACA": ml_dataset['ICI_IRM_NONLAC_D_ACA'],
                    "ICI_IRM_NONLAC_D_ACP": ml_dataset['ICI_IRM_NONLAC_D_ACP'],
                    "ICI_IRM_NONLAC_D_ACHA": ml_dataset['ICI_IRM_NONLAC_D_ACHA'],
                    "ICI_IRM_NONLAC_D_IPP": ml_dataset['ICI_IRM_NONLAC_D_IPP'],
                    "ICI_IRM_NONLAC_D_LB": ml_dataset['ICI_IRM_NONLAC_D_LB'],
                    "ICI_IRM_NONLAC_D_AITC": ml_dataset['ICI_IRM_NONLAC_D_AITC'],
                    "ICI_IRM_NONLAC_D_ACPI": ml_dataset['ICI_IRM_NONLAC_D_ACPI'],
                    "ICI_IRM_NONLAC_D_ACAI": ml_dataset['ICI_IRM_NONLAC_D_ACAI'],
                    "ICI_IRM_NONLAC_D_ACS": ml_dataset['ICI_IRM_NONLAC_D_ACS'],
                    "ICI_IRM_NONLAC_G_ACM": ml_dataset['ICI_IRM_NONLAC_G_ACM'],
                    "ICI_IRM_NONLAC_G_ACA": ml_dataset['ICI_IRM_NONLAC_G_ACA'],
                    "ICI_IRM_NONLAC_G_ACP": ml_dataset['ICI_IRM_NONLAC_G_ACP'],
                    "ICI_IRM_NONLAC_G_ACHA": ml_dataset['ICI_IRM_NONLAC_G_ACHA'],
                    "ICI_IRM_NONLAC_G_IPP": ml_dataset['ICI_IRM_NONLAC_G_IPP'],
                    "ICI_IRM_NONLAC_G_LB": ml_dataset['ICI_IRM_NONLAC_G_LB'],
                    "ICI_IRM_NONLAC_G_AITC": ml_dataset['ICI_IRM_NONLAC_G_AITC'],
                    "ICI_IRM_NONLAC_G_ACPI": ml_dataset['ICI_IRM_NONLAC_G_ACPI'],
                    "ICI_IRM_NONLAC_G_ACAI": ml_dataset['ICI_IRM_NONLAC_G_ACAI'],
                    "ICI_IRM_NONLAC_G_ACS": ml_dataset['ICI_IRM_NONLAC_G_ACS'],
                    "OG_ETAT": ml_dataset['OG_ETAT'],
                    
                    }

                return pd.DataFrame(patient_data, index=[0])

            # Load the patient data
            df_patient_data = load_data()
            t_df = df_patient_data.T

            # Display a progress bar while the operation is in progress
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)

            # Save the new dataset
            path = r"./streamlit/data/patient_database/{}.csv".format(patient_index)
            df_patient_data.to_csv(path, index=False)
            st.success("New dataset saved !", icon="✅")
            st.cache_data.clear()

            # Display results and model info in expanders
            with st.expander("Results"):
                df = pd.read_csv("./streamlit/data/patient_database/{}.csv".format(patient_index))
                st.markdown("## Results")
                st.dataframe(t_df, use_container_width=True)

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