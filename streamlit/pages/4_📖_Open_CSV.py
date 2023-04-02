import streamlit as st
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import xgboost as xgb

######################################### Functions ###################################################

def normalize_column(column_name: str):
    return str(column_name.replace(' ', '_') \
                        .replace('é', 'e') \
                        .replace('à', 'a') \
                        .replace("'", '_').upper())

def load_data(patient_index):
    path = r"./streamlit/data/patient_database/{}.csv".format(patient_index)
    data = pd.read_csv(path)
    return data

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

########################################## Streamlit ################################################################


st.title("Open patient data and related informations")

st.subheader("Choose the way to open the patient data")

st.session_state['mode'] = ""

col1, col2 = st.columns(2)

with col1:

    patient_index = st.text_input("Patient index",placeholder= "Enter patient index")

    if st.button("Show from index"):
        if patient_index is not None:
            st.session_state['mode'] = "index"
            st.session_state['patient_index'] = patient_index
            

with col2:
    patient_file = st.file_uploader("Upload patient file", type="csv")
    if patient_file is not None:
        data = pd.read_csv(patient_file)

        ml_dataset = data_preprocessing(data)
        
        @st.cache_data
        def load_csv():
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

    
        with patient_file:
            if st.button("Show from CSV file"):
                if patient_file is not None:
                    st.session_state['mode'] = "file"


if st.session_state['mode'] == "index":
    tab1, tab2 = st.tabs(["Patient data", "Model Metrics"])
    st.session_state['mode'] = ""
    with tab1:
        patient_index = st.session_state['patient_index']
        df = load_data(patient_index)
        st.dataframe(df.T, use_container_width=True)
    with tab2:
        st.markdown("Model metrics")
        arr = np.random.normal(1, 1, size=100)
        fig, ax = plt.subplots()
        ax.hist(arr, bins=20)
        st.pyplot(fig)
    
elif st.session_state['mode'] == "file":
    if patient_file is not None:
        st.session_state['mode'] = ""
        tab1, tab2 = st.tabs(["Patient data", "Model Metrics"])
        with tab1:
            df = load_csv()
            st.dataframe(df.T, use_container_width=True)
        with tab2:
            st.markdown("Model metrics")
            arr = np.random.normal(1, 1, size=100)
            fig, ax = plt.subplots()
            ax.hist(arr, bins=20)
            st.pyplot(fig)

