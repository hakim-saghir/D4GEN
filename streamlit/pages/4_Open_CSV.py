import streamlit as st
import pandas as pd
import time
import os


def normalize_column(column_name: str):
    return str(column_name.replace(' ', '_') \
                        .replace('é', 'e') \
                        .replace('à', 'a') \
                        .replace("'", '_').upper())

def load_data(patient_index):
    path = r"./data/patient_database/{}.csv".format(patient_index)
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
        [
            'AGE', 'ATCD_CONSO_ALCOOL', 'ATCD_DIABETE', 'ATCD_FA_CONNU', 'ATCD_HTA', 'BIO_CPK',
            'BIO_CRP', 'BIO_LDL', 'BIO_NTPROBNP', 'BIO_POTASSIUM', 'BIO_HBA1C', 'BIO_TROPONINE', 'BIO_TSH3G',
            'CDS_EE_AUTRE', 'CDS_EE_AUTRE_TEXT', 'CDS_EE_ETO', 'CDS_EE_ETT', 'CDS_EE_HECGTYPE', 'ECG_A_L_ARRIVEE',
            'ECPL_BIO_NTPROBNP', 'ECPL_ETT', 'ECPL_HOLTER', 'ECPL_SCOP',
            'ETIO_CE_FA', 'CDS_EE_HECG', 'ETIO_TOAST', 'FA_SUR_ECGSCOPEHOLTERREVEAL', 'ICI_ASPECT', 'ICI_IRM_NONLAC_D_ACM',
            'ICI_SWAN_THROMBUS', 'NIHSS_INITIAL', 'ICI_IRM_LAC', 'SEXE', 'UNITE_ALCOOL/SEM'
        ]
    ]

    return ml_dataset

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
        
elif st.session_state['mode'] == "file":
    if patient_file is not None:
        st.session_state['mode'] = ""
        tab1, tab2 = st.tabs(["Patient data", "Model Metrics"])
        with tab1:
            df = load_csv()
            st.dataframe(df.T, use_container_width=True)
        with tab2:
            st.markdown("Model metrics")
        

