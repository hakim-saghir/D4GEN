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
        [
            'AGE', 'BIO_CPK',
            'BIO_CRP', 'BIO_LDL', 'BIO_NTPROBNP', 'BIO_POTASSIUM', 'BIO_HBA1C', 'BIO_TROPONINE', 'BIO_TSH3G',
            'CDS_EE_AUTRE', 'CDS_EE_AUTRE_TEXT', 'CDS_EE_ETO', 'CDS_EE_ETT', 'CDS_EE_HECGTYPE', 'ECG_A_L_ARRIVEE',
            'ECPL_BIO_NTPROBNP', 'ECPL_ETT', 'ECPL_HOLTER', 'ECPL_SCOP',
            'ETIO_CE_FA', 'CDS_EE_HECG', 'ETIO_TOAST', 'FA_SUR_ECGSCOPEHOLTERREVEAL', 'ICI_ASPECT', 'ICI_IRM_NONLAC_D_ACM',
            'ICI_SWAN_THROMBUS', 'NIHSS_INITIAL', 'ICI_IRM_LAC', 'SEXE', 'UNITE_ALCOOL/SEM'
        ]
    ]
    st.markdown(len(ml_dataset))

    return ml_dataset

# Function to check if the uploaded file has the correct format
def check_data(csv_file):
    try:
        df = pd.read_csv(csv_file)  # Import the csv file

        if df.shape[1] != 18:
            st.write(df.shape)
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
                    "SEXE": ml_dataset['SEXE'],
                    "ATCD_HTA": ml_dataset['ATCD_HTA'],
                    "UNITE_ALCOOL/SEM": ml_dataset['UNITE_ALCOOL/SEM'],
                    "ATCD_CONSO_ALCOOL": ml_dataset['ATCD_CONSO_ALCOOL'],
                    "ATCD_FA_CONNU": ml_dataset['ATCD_FA_CONNU'],
                    "NIHSS_INITIAL": ml_dataset['NIHSS_INITIAL'],
                    "ICI_IRM_LAC": ml_dataset['ICI_IRM_LAC'],
                    "ICI_IRM_NONLAC_D_ACM": ml_dataset['ICI_IRM_NONLAC_D_ACM'],
                    "ICI_ASPECT": ml_dataset['ICI_ASPECT'],
                    "ICI_SWAN_THROMBUS": ml_dataset['ICI_SWAN_THROMBUS'],
                    "ECG_A_L_ARRIVEE": ml_dataset['ECG_A_L_ARRIVEE'],
                    "ECPL_ETT": ml_dataset['ECPL_ETT'],
                    "ECPL_BIO_NTPROBNP": ml_dataset['ECPL_BIO_NTPROBNP'],
                    "BIO_NTPROBNP": ml_dataset['BIO_NTPROBNP'],
                    "ETIO_TOAST": ml_dataset['ETIO_TOAST'],
                    "BIO_POTASSIUM": ml_dataset['BIO_POTASSIUM'],
                    "BIO_HBA1C": ml_dataset['BIO_HBA1C'],
                    "BIO_LDL": ml_dataset['BIO_LDL'],
                    "BIO_CPK": ml_dataset['BIO_CPK'],
                    "BIO_TROPONINE": ml_dataset['BIO_TROPONINE'],
                    "BIO_CRP": ml_dataset['BIO_CRP'],
                    "BIO_TSH3G": ml_dataset['BIO_TSH3G'],
                    "ATCD_DIABETE": ml_dataset['ATCD_DIABETE'],
                    "ECPL_HOLTER": ml_dataset['ECPL_HOLTER'],
                    "ECPL_SCOP": ml_dataset['ECPL_SCOP'],
                    "ETIO_CE_FA": ml_dataset['ETIO_CE_FA'],
                    "CDS_EE_HECG": ml_dataset['CDS_EE_HECG'],
                    "CDS_EE_HECGTYPE": ml_dataset['CDS_EE_HECGTYPE'],
                    "FA_SUR_ECGSCOPEHOLTERREVEAL": ml_dataset['FA_SUR_ECGSCOPEHOLTERREVEAL'],
                    "CDS_EE_AUTRE_TEXT": ml_dataset['CDS_EE_AUTRE_TEXT'],
                    "CDS_EE_AUTRE": ml_dataset['CDS_EE_AUTRE'],
                    "CDS_EE_ETO": ml_dataset['CDS_EE_ETO'],
                    "CDS_EE_ETT": ml_dataset['CDS_EE_ETT'],
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
            path = r"./data/patient_database/{}.csv".format(patient_index)
            df_patient_data.to_csv(path, index=False)
            st.success("New dataset saved !", icon="✅")
            st.cache_data.clear()

            # Display results and model info in expanders
            with st.expander("Results"):
                df = pd.read_csv("data/patient_database/{}.csv".format(patient_index))
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