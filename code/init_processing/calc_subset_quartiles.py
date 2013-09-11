##### SETUP ######

import aggregate
import pickle
import numpy as np
import pandas as pd

percentiles = np.array([5, 25, 50, 75, 95])

##################

selection_age = 5
age_greater_than = True
until_age = 8

bmi_greater_than = 17.3
bmi_less_than = np.nan

##################

df = pickle.load(open('../../data/pkl/BMI_resampled.pkl', 'rb'))

## Restrict to patients who satisfy start criteria
grouped = df.groupby("id")

for name_patient, group_patient in grouped:
    patient_age = group_patient["age"]
    if patient_age.min() > selection_age or patient_age.max() < selection_age:
        df[df["id"] == name_patient] = np.nan
    elif ~np.isnan(bmi_less_than) and ~np.isnan(bmi_greater_than):
        if group_patient[group_patient["age"] == selection_age]["bmi"] < bmi_greater_than or \
           group_patient[group_patient["age"] == selection_age]["bmi"] > bmi_less_than:
            df[df["id"] == name_patient] = np.nan
    elif ~np.isnan(bmi_less_than):
        if group_patient[group_patient["age"] == selection_age]["bmi"] > bmi_less_than:
            df[df["id"] == name_patient] = np.nan
    elif ~np.isnan(bmi_greater_than):
        if group_patient[group_patient["age"] == selection_age]["bmi"] < bmi_greater_than:
            df[df["id"] == name_patient] = np.nan
df = df.dropna()

if age_greater_than:
    df = df[df["age"] <= selection_age]
else:
    df = df[df["age"] >= selection_age]

## Group datapoints by gender and age
groupby_attributes = ["gender","age"]

## Calculate aggregate values
df_aggregate = aggregate.calculate_aggregations(df, groupby_attributes, percentiles)

if age_greater_than:
    df_aggregate = df_aggregate[df_aggregate["age"] < until_age]
else:
    df_aggregate = df_aggregate[df_aggregate["age"] > until_age]

## Save dataframes to pickle
output = open('../../data/pkl/BMI_subset_aggregate.pkl', 'wb')
pickle.dump(df_aggregate, output, -1)
output.close()
output = open('../../data/pkl/BMI_subset_resampled.pkl', 'wb')
pickle.dump(df, output, -1)
output.close()

## Save dataframes to CSV
df.to_csv("../../data/csv/BMI_subset_resampled.csv", index_label=False, index=False)
df_aggregate.to_csv("../../data/csv/BMI_subset_aggregate.csv", index_label=False, index=False)
