##### SETUP ######

import sys
sys.path.append('../config')

import config
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

header = config.header
header_list = config.header_list

attributes = config.attributes
percentiles = config.percentiles

##################

## Open pickle file, saved from bmi_initial_processing.py
pkl_file = open('../../data/pkl/bmi_data.pkl', 'rb')

df_resampled = pickle.load(pkl_file)
df_filtered = pickle.load(pkl_file)
df_filtered_out = pickle.load(pkl_file)
df_outliers = pickle.load(pkl_file)

df = df_filtered

grouped = df.groupby([header["id"]])

for name_patient, group_patient in grouped:
    patient_age = group_patient[header["age"]]
    if not (patient_age.min() < 4 and patient_age.max() > 8):
        patient_id = group_patient[header["id"]].iloc[0]
        df[df[header["id"]] == patient_id] = np.nan

df_ranged = df.dropna()

output = open('../../data/pkl/bmi_data_ar.pkl', 'wb')
pickle.dump(df_ranged, output, -1)

output.close()

df_resampled.to_csv("../../data/csv/BMI_resampled_ar_5dpt.csv")
