##### SETUP ######

import sys
sys.path.append('../config')

import config
import aggregate

import pickle
import numpy as np
import pandas as pd

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
    
df = df_resampled

## Only one individual is a PI/HN; thus, take this patient out of the dataset
## To find: df_filtered[df_filtered[header_race_ethnicity] == "Pacific Islander/Hawaiian Native"]
df[df[header["race_ethnicity"]] == "Pacific Islander/Hawaiian Native"] = np.nan
df.dropna()

## Initialize new DataFrame for aggregated information
df_aggregate = pd.DataFrame(columns = header_list)

## Group patients by gender, race/ethnicity, and age groups
header_list_groupby = [header["gender"],header["age"],header["race_ethnicity"]]
## Calculate aggregate values
df_aggregate = aggregate.calculate_aggregations(df_aggregate, df, header_list_groupby, header_list, percentiles)

## Repeats the above calculation, except all races/ethnicities are lumped into one category, “All” 
## Group patients by gender and age
header_list_groupby = [header["gender"],header["age"]]
## Calculate aggregate values
df_aggregate = aggregate.calculate_aggregations(df_aggregate, df, header_list_groupby, header_list, percentiles)
    
## Save aggregate DataFrame to pickle
output = open('../../data/pkl/bmi_data_aggregate.pkl', 'wb')
pickle.dump(df_aggregate, output, -1)
output.close()

## Save aggregate DataFrame to CSV
df_aggregate.to_csv("../../data/csv/BMI_aggregate_allyrs_5dpt.csv", index_label=False, index=False)
