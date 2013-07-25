import sys
sys.path.append('../config')

import config
import numpy as np
import pandas as pd

header = config.header
attributes = config.attributes

def calculate_aggregations(df_aggregate, df, header_list_groupby, header_list, percentiles):
    grouped = df.groupby(header_list_groupby)

    percentiles_int, num_percentiles = config.convert_percentiles(percentiles)
    
    for name, group in grouped:
        ## Create new numpy list that is of the correct dimensions
        row = np.array([None]*len(header_list))
        row = np.reshape(row, (1, len(header_list)))
        
        ## Create new DataFrame row with correct column headers 
        df_row = pd.DataFrame(row, columns=header_list)

        ## Compute/save relevant information
        df_row[header["gender"]] = name[0]      
        df_row[header["age"]] = name[1]

        if header["race_ethnicity"] in header_list_groupby:
            df_row[header["race_ethnicity"]] = name[2]
        else:
            df_row[header["race_ethnicity"]] = "All"

        num_patients = group.shape[0]
        percentile_indices = np.round(percentiles*num_patients).astype(int).tolist()
        percentile_indices = [num_patients - 1 if x == num_patients else x for x in percentile_indices]
        
        for attribute in attributes.keys():
            group = group.sort([header[attribute]])
            percentile_values = [group.iloc[x] for x in percentile_indices]

            df_row[header[attribute + "_mean"]] = group[header[attribute]].mean()
            df_row[header[attribute + "_std"]] = group[header[attribute]].std()
            for index in range(num_percentiles):
                df_row[header[attribute + "_" + str(percentiles_int[index])]] = percentile_values[index][header[attribute]]
       
        df_row[header["count"]] = num_patients
        
        ## Append new DataFrame row into aggregate DataFrame
        df_aggregate = pd.DataFrame.append(df_aggregate, df_row)

    return df_aggregate
