##### SETUP ######

import sys
sys.path.append('../config')

import config
import pickle
import numpy as np
import pandas as pd
import math
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import pylab as P

## Age points to resample data from
intervals = config.intervals

## Column headers
header = config.header

##################

##### VARIABLES #####

## Minimum number of datapoints for patient to pass filter
min_datapoints = 5

## CSV raw document locaiton
csv_location = "../../data/csv/BMI.csv"

## Individuals with outlier data
## 16262: First two datapoints when ages ~1.8 have heights of ~2 inches
bad_patients = [16262]

#####################

##### FUNCTIONS #####

## Returns a new DataFrame with same headers and correct dtypes as input DataFrame

## Change numeric datatypes to appropriate floats of ints
def change_dtypes(df):

    if header["wt"] in df.columns:
        df[header["wt"]] = df[header["wt"]].astype(np.float64)
    else:
        df[header["wt_ounces"]] = df[header["wt_ounces"]].astype(np.float64)

    df[header["age"]] = df[header["age"]].astype(np.float64)
    df[header["bmi"]] = df[header["bmi"]].astype(np.float64)
    df[header["id"]] = df[header["id"]].astype(np.int64)
    df[header["ht"]] = df[header["ht"]].astype(np.float64)
    return df

## Returns a new DataFrame with same headers and correct dtypes as input DataFrame
def create_new_df_with_columns(df):
    import pandas as pd
    
    df_new = pd.DataFrame(columns=df.columns)
    df_new = change_dtypes(df_new)
    return df_new

## Fit a OLS curve to the group patient's y as a function of x
def fit(group_patient, header_y, header_x, alpha = 0.01):
    import numpy as np
    import statsmodels.api as sm
    from scipy import stats 
 
    x = group_patient[header_x]
    y = group_patient[header_y]
    num_x = group_patient.shape[0]
    ## np.ones necessary to include constant in regression
    X = np.column_stack((x*x*x, x*x, x, np.ones(num_x)))
    ## OLS fit
    model = sm.OLS(y,X)
    results = model.fit()

    infl = results.get_influence()
    resids = infl.resid_studentized_external
    resid_cutoff = stats.t.ppf(1.-alpha/2, results.df_resid)
    large_resid = np.abs(resids) > resid_cutoff

    (c, p) = infl.cooks_distance
    cooks_cutoff = 4.0/group_patient.shape[0]
    large_cooks = c > cooks_cutoff
    
    return large_resid #np.logical_or(large_cooks, large_resid)

## Fit a OLS curve to the group patient's y as a function of x
def fit2(group_patient, header_y, header_x):
    import numpy as np
    import statsmodels.api as sm
 
    x = group_patient[header_x]
    y = group_patient[header_y]
    num_x = group_patient.shape[0]
    ## np.ones necessary to include constant in regression
    X = np.column_stack((np.log(x), np.ones(num_x)))
    ## OLS fit
    model = sm.OLS(y,X)
    fitted = model.fit()
    #print fitted.summary()
    #influence = fitted.get_influence()
    #(c, p) = influence.cooks_distance

    ## Outliers defined to be either datapoints that are greater than 2.5 std
    outliers_bool_norm_resid = abs(fitted.norm_resid()) > 2.5

    ## or greater than 4/(N-K-1)
    #outliers_bool_cooks_distance = c > 4.0/(num_x-2)

    #outliers_bool = elementwise_or(outliers_bool_norm_resid, outliers_bool_cooks_distance)
    
    return outliers_bool_norm_resid

## OR function, element by element for lists of the same length
def elementwise_or(*args):
    return [any(tuple) for tuple in zip(*args)]

#####################

## Read in document
df = pd.read_csv(csv_location)
#df = df[1:50000]

## Drop columns not needed
df = df.drop(["Height (ft&inc)","Body Mass Index Percentile","Age on 5/1/2012","BMI","Patient Ethnicity","Patient Race"], axis=1)

## Convert numerical columns to floats (and remove #VALUE! if necessary)
cond = df[header["ht"]] == "#VALUE!"
df[header["ht"]][cond] = "NaN"
df = change_dtypes(df)

## Fill in missing values: need at least one of bmi, wt, ht to resolve third
selector = np.isnan(df[header["bmi"]]) & ~np.isnan(df[header["wt_ounces"]]) & ~np.isnan(df[header["ht"]])
missing_series = df[selector][header["wt_ounces"]]/16/(np.power(df[selector][header["ht"]],2))*703
df[header["bmi"]] = pd.concat([df[header["bmi"]].dropna(), missing_series.dropna()]).reindex_like(df)

selector = ~np.isnan(df[header["bmi"]]) & np.isnan(df[header["wt_ounces"]]) & ~np.isnan(df[header["ht"]])
missing_series = df[selector][header["bmi"]]*np.power(df[selector][header["ht"]],2)/703*16
df[header["wt"]] = pd.concat([df[header["wt_ounces"]].dropna(), missing_series.dropna()]).reindex_like(df)

selector = ~np.isnan(df[header["bmi"]]) & ~np.isnan(df[header["wt_ounces"]]) & np.isnan(df[header["ht"]])
missing_series = np.power(df[selector][header["wt_ounces"]]/16/(df[selector][header["bmi"]])*703,0.5)
df[header["ht"]] = pd.concat([df[header["ht"]].dropna(), missing_series.dropna()]).reindex_like(df)

## Remove rows that can't be resolved for bmi, wt, or ht
selector = ~np.isnan(df[header["bmi"]]) & ~np.isnan(df[header["wt_ounces"]]) & ~np.isnan(df[header["ht"]]) & df[header["gender"]].notnull()
df = df[selector]

## Convert from ounces to pounds
df[header["wt"]] = df[header["wt_ounces"]]/16
df = df.drop(header["wt_ounces"], axis=1)

## Remove bad patients
for patient_id in bad_patients:
    df[df[header["id"]] == patient_id] = np.nan
df = df.dropna()
    
## Sort columns by header id and reindex
df = df.sort([header["id"], header["age"]], ascending=[1,1])
df = df.reindex()

## Group by patient id
grouped_patients = df.groupby([header["id"]])

## Initialize DataFrames
df_resampled = create_new_df_with_columns(df)
df_filtered = create_new_df_with_columns(df)
df_filtered_out = create_new_df_with_columns(df)
df_outliers = create_new_df_with_columns(df)

## Iterate through each patient in the dataframe
for name_patient, group_patient in grouped_patients:
    ## To grab patients, type:
    #grouped_patients.get_group(np.int64(5035))

    ## Skip patients that have less than minimum number of datapoints
    if group_patient.shape[0] < min_datapoints:
        df_filtered_out = pd.DataFrame.append(df_filtered_out, group_patient)
        continue

    ## Change datapoints with age 0 to age 0.01 (in order to take its log for the regression)
    group_patient[header["age"]][group_patient[header["age"]] == np.float64(0)] = np.float64(0.01)

    ## Conduct regression of ht vs age and wt vs age
    if group_patient[header["age"]].min() > 15:
        outliers_bool_ht = fit2(group_patient, header["ht"], header["age"])
        outliers_bool_wt = fit2(group_patient, header["wt"], header["age"])
    else:
        outliers_bool_ht = fit(group_patient, header["ht"], header["age"])
        outliers_bool_wt = fit(group_patient, header["wt"], header["age"])
    outliers_bool = elementwise_or(outliers_bool_ht, outliers_bool_wt)

    ## Remove outlier datapoints and save removed datapoints to df_outliers
    if any(outliers_bool):
        df_outliers = pd.DataFrame.append(df_outliers, group_patient[outliers_bool])
    group_patient = group_patient[np.logical_not(outliers_bool)]

    ## Interpolate ht and wt linearly and resample at defined age intervals
    f_ht = interpolate.interp1d(group_patient[header["age"]], group_patient[header["ht"]], kind='linear', bounds_error=False)
    f_wt = interpolate.interp1d(group_patient[header["age"]], group_patient[header["wt"]], kind='linear', bounds_error=False)
    df_resampled_patient = pd.DataFrame(intervals, columns=[header["age"]])
    df_resampled_patient[header["ht"]] = df_resampled_patient[header["age"]].apply(f_ht)
    df_resampled_patient[header["wt"]] = df_resampled_patient[header["age"]].apply(f_wt)

    ## Drop NAs, which occur because no extrapolation is conducted
    df_resampled_patient = df_resampled_patient.dropna()

    ## Recalculate BMIs
    df_resampled_patient[header["bmi"]] = df_resampled_patient[header["wt"]]/16/(np.power(df_resampled_patient[header["ht"]],2))*703

    ## Save patient attributes
    df_resampled_patient[header["id"]] = name_patient
    df_resampled_patient[header["gender"]] = group_patient[header["gender"]].max()
    df_resampled_patient[header["race_ethnicity"]] = group_patient[header["race_ethnicity"]].max()

    ## Save filtered and resampled data
    df_resampled = pd.DataFrame.append(df_resampled, df_resampled_patient)
    df_filtered = pd.DataFrame.append(df_filtered, group_patient)

## Print dataframes to CSVs
df_resampled.to_csv("../../data/csv/BMI_resampled_allyrs_5dpt.csv", index_label=False, index=False)
df_filtered.to_csv("../../data/csv/BMI_filtered_allyrs_5dpt.csv", index_label=False, index=False)
df_filtered_out.to_csv("../../data/csv/BMI_filtered_out_allyrs_5dpt.csv", index_label=False, index=False)
df_outliers.to_csv("../../data/csv/BMI_outliers_allyrs_5dpt.csv", index_label=False, index=False)

## Dump files with pickle using the highest protocol available
output = open('../../data/pkl/bmi_data.pkl', 'wb')
pickle.dump(df_resampled, output, -1)
pickle.dump(df_filtered, output, -1)
pickle.dump(df_filtered_out, output, -1)
pickle.dump(df_outliers, output, -1)

output.close()
