##### SETUP ######

import pickle
import numpy as np
import pandas as pd

##### VARIABLES ######

contained_age = 5

##################

## Open pickle file, saved from bmi_initial_processing.py
df = pickle.load(open('../../data/pkl/BMI_filtered.pkl', 'rb'))

grouped = df.groupby(["id"])

for name_patient, group_patient in grouped:
    patient_age = group_patient["age"]
    if not (patient_age.min() < contained_age and patient_age.max() > contained_age):
        patient_id = group_patient["id"].iloc[0]
        df[df["id"] == patient_id] = np.nan

df_ranged = df.dropna()

output = open('../../data/pkl/BMI_filtered_contain_age5.pkl', 'wb')
pickle.dump(df_ranged, output, -1)
output.close()

