##### SETUP ######

import sys
sys.path.append('../config')

import config
import aggregate

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

header = config.header
header_list = config.header_list

attributes = {"bmi":"BMI"} #config.attributes

percentiles = np.array([.05, .33, .66, .8])
header_list = config.add_attributes_to_header_dict(attributes, header, header_list, percentiles)

##################

pkl_file = open('../../data/pkl/bmi_data_expansion.pkl', 'rb')
df_expansion = pickle.load(pkl_file)

df = df_expansion

## Restrict to patients who satisfy start criteria
grouped = df.groupby(header["id"])
for name_patient, group_patient in grouped:
    patient_age = group_patient[header["age"]]
    if patient_age.min() >5 or patient_age.max() < 5:
        patient_id = group_patient[header["id"]].iloc[0]
        df[df[header["id"]] == patient_id] = np.nan
    elif group_patient[group_patient[header["age"]] == 5.0][header["bmi"]] < 17.4:
        patient_id = group_patient[header["id"]].iloc[0]
        df[df[header["id"]] == patient_id] = np.nan
            
df[df[header["age"]] < 5.0] = np.nan
df = df.dropna()

## Calculate percentiles
header_list_groupby = [header["gender"],header["age"]]

## Initialize new DataFrame for aggregated information
df_aggregate = pd.DataFrame(columns = header_list)

## Group patients by gender, race/ethnicity, and age groups
header_list_groupby = [header["gender"],header["age"]]

## Calculate aggregate values
df_aggregate = aggregate.calculate_aggregations(df_aggregate, df, header_list_groupby, header_list, percentiles)

df_aggregate[df_aggregate[header["age"]] > 10.0] = np.nan
df_aggregate = df_aggregate.dropna()

## Save aggregate DataFrame to pickle
output = open('../../data/pkl/bmi_data_expansion_aggregate.pkl', 'wb')
pickle.dump(df_aggregate, output, -1)
output.close()
output = open('../../data/pkl/bmi_data_expansion.pkl', 'wb')
pickle.dump(df, output, -1)
output.close()

## Save aggregate DataFrame to CSV
df_resampled.to_csv("../../data/csv/BMI_resampled_expansion_age5_bmi_gr_17dot4.csv", index_label=False, index=False)
df_aggregate.to_csv("../../data/csv/BMI_resampled_expansion_aggregate_age5_bmi_gr_17dot4.csv", index_label=False, index=False)
