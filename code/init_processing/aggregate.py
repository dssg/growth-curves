import numpy as np
import pandas as pd

def calculate_aggregations(df, groupby_attributes, percentiles):

    ## Group datapoints by interested attributes
    grouped = df.groupby(groupby_attributes)

    ## Attributes to calculate percentiles for
    attributes = ["ht", "wt", "bmi"]

    ## Create list of dictionaries to append to
    df_row_list = []

    ## Iterate through each group
    for name, group in grouped:
        print name

        ## Create new dictionary
        df_row = dict()
        
        ## Compute/save relevant information
        df_row["gender"] = name[0]      
        df_row["age"] = name[1]

        ## If race/ethnicity is not a groupby attribute, calculate percentiles for all races/ethnicities lumped together
        ## under "All"
        if "race_ethnicity" in groupby_attributes:
            df_row["race_ethnicity"] = name[2]
        else:
            df_row["race_ethnicity"] = "All"

        ## Compute number of patients in group
        num_patients = group.shape[0]
        df_row["count"] = num_patients

        ## Determine indices for each percentile
        percentile_indices = np.round(percentiles/100.0*num_patients).astype(int).tolist()

        ## If a percentile index is equal to the number of patients, reduce it by 1 so that it is a valid index
        ## (sinces indices go from 0 to num_patients - 1)
        percentile_indices = [num_patients - 1 if x == num_patients else x for x in percentile_indices]

        for attribute in attributes:
            ## Groupby attribute
            group = group.sort([attribute])

            ## Retrive percentile values
            percentile_values = [group.iloc[x] for x in percentile_indices]

            ## And save these percentile values
            for index in range(len(percentiles)):
                df_row[attribute + "_" + str(percentiles[index])] = percentile_values[index][attribute]
                
            ## Compute means and standard deivations
            df_row[attribute + "_mean"] = group[attribute].mean()
            df_row[attribute + "_std"] = group[attribute].std()

        ## Append new dictionary to list
        df_row_list.append(df_row)

    ## Return dataframe from list of dictionaries
    return pd.DataFrame(df_row_list)
