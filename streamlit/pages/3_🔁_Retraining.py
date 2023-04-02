import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

######################################### Functions ###################################################

def data_preprocessing(csv_file):

    prepared_dataset = csv_file.copy()

    # Elimination des lignes/colonnes vides
    prepared_dataset = prepared_dataset.dropna(how='all')

    def normalize_column(column_name: str):
        return str(column_name.replace(' ', '_')
                   .replace('é', 'e')
                   .replace('à', 'a')
                   .replace("'", '_').upper())

    # Normalisation des noms des colonnes
    for i, c in enumerate(prepared_dataset.columns):
        prepared_dataset = prepared_dataset.rename(
            columns={c: normalize_column(c)})

    # Selection des colonnes intéréssantes
    # A ajouter ECPL_ETT_TXT, ECPL_SCOP_TXT, ECPL_HOLTER_TXT


    ml_dataset = prepared_dataset[columns]

    print(ml_dataset.columns)

    return ml_dataset

st.title("Retraining")
st.subheader("Retraining the model with new data")

base_dataset = pd.read_csv("./streamlit/data/dataset/prepared/raw_dataset_v2.csv", sep=';')
uploaded_file = st.file_uploader("Upload CSV", type="csv")



columns = ['AGE', 'ATCD_CONSO_ALCOOL', 'ATCD_DIABETE', 'ATCD_HTA', 'BIO_CPK',
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
       'ICI_IRM_NONLAC_G_ACAI', 'ICI_IRM_NONLAC_G_ACS']
    

###########################################################################################################################



if uploaded_file is not None:

    df_patient_data = pd.read_csv(uploaded_file)

    base_dataset = data_preprocessing(base_dataset)
    df_patient_data = data_preprocessing(df_patient_data)

    base_dataset = base_dataset.append(
        dict(zip(columns, df_patient_data.iloc[0].tolist())), ignore_index=True)
    with uploaded_file:
        if st.button("Save new dataset !"):

            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)

            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            path = r"./streamlit/data/dataset/appendedDataset.csv"
            base_dataset.to_csv(path, index=False)
            st.success("New dataset saved !", icon="✅")
            st.cache_data.clear()