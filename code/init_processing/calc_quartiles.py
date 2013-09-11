##### SETUP ######

import aggregate

import pickle
import numpy as np
import pandas as pd

##################

##### VARIABLES #####

percentiles = np.array([3, 5, 10, 25, 50, 75, 85, 90, 95, 97])

#####################

## Open pickle file, saved from bmi_initial_processing.py
df = pickle.load(open('../../data/pkl/BMI_resampled_lin.pkl', 'rb'))

## Only one individual is a PI/HN; thus, take this patient out of the dataset
df[df["race_ethnicity"] == "Pacific Islander/Hawaiian Native"] = np.nan
df.dropna()

## Group datapoints by gender, race/ethnicity, and age
groupby_attributes = ["gender","age","race_ethnicity"]
## Calculate aggregate values
df_aggregate = aggregate.calculate_aggregations(df, groupby_attributes, percentiles)

## Repeats the above calculation, except all races/ethnicities are lumped into one category, “All”

## Group datapoints by gender and age
groupby_attributes = ["gender","age"]
## Calculate aggregate values
df_aggregate = df_aggregate.append(aggregate.calculate_aggregations(df, groupby_attributes, percentiles))
    
## Save aggregate DataFrame to pickle
output = open('../../data/pkl/BMI_aggregate_percentiles.pkl', 'wb')
pickle.dump(df_aggregate, output, -1)
output.close()

## Save aggregate DataFrame to CSV
df_aggregate.to_csv("../../data/csv/BMI_aggregate_percentiles.csv", index_label=False, index=False)
