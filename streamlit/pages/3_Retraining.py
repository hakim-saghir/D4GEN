import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

st.title("Retraining")
st.subheader("Retraining the model with new data")

base_dataset = pd.read_csv('../data/raw/raw_dataset.csv', sep=';')
uploaded_file = st.file_uploader("Upload CSV", type="csv")



columns = [
    'AGE', 'ATCD_CONSO_ALCOOL', 'ATCD_DIABETE', 'ATCD_FA_CONNU', 'ATCD_HTA', 'BIO_CPK',
    'BIO_CRP', 'BIO_LDL', 'BIO_NTPROBNP', 'BIO_POTASSIUM', 'BIO_HBA1C', 'BIO_TROPONINE', 'BIO_TSH3G',
    'CDS_EE_AUTRE', 'CDS_EE_AUTRE_TEXT', 'CDS_EE_ETO', 'CDS_EE_ETT', 'CDS_EE_HECGTYPE', 'ECG_A_L_ARRIVEE',
    'ECPL_BIO_NTPROBNP', 'ECPL_ETT', 'ECPL_HOLTER', 'ECPL_SCOP',
    'ETIO_CE_FA', 'CDS_EE_HECG', 'ETIO_TOAST', 'FA_SUR_ECGSCOPEHOLTERREVEAL', 'ICI_ASPECT', 'ICI_IRM_NONLAC_D_ACM',
    'ICI_SWAN_THROMBUS', 'NIHSS_INITIAL', 'ICI_IRM_LAC', 'SEXE', 'UNITE_ALCOOL/SEM'
]


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
            path = r"./data/dataset/appendedDataset.csv"
            base_dataset.to_csv(path, index=False)
            st.success("New dataset saved !", icon="✅")
            st.balloons()
            st.cache_data.clear()