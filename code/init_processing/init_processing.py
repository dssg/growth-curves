##### SETUP ######

import pickle
import numpy as np
import pandas as pd
import math
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import pylab as P
import numpy as np
import statsmodels.api as sm
from scipy import stats 
    
## Set up age intervals
## Float instability created errors, so corrected the following trunction 
intervals = np.trunc(np.concatenate((np.array([0.01]),np.arange(.05,.2,.05),np.arange(.4,2,.2), np.arange(2,20,.5)))*100)/100

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

## Change numeric datatypes to appropriate floats of ints
def change_dtypes(df):
    if "wt" in df.columns:
        df["wt"] = df["wt"].astype(np.float64)
    else:
        df["wt_ounces"] = df["wt_ounces"].astype(np.float64)
    df["age"] = df["age"].astype(np.float64)
    df["bmi"] = df["bmi"].astype(np.float64)
    df["id"] = df["id"].astype(np.int64)
    df["ht"] = df["ht"].astype(np.float64)
    return df

## Fit a OLS curve to the group patient's y as a function of x
def fit(group_patient, header_y, header_x, alpha = 0.01):
    
    x = group_patient[header_x]
    y = np.log(group_patient[header_y])

    ## Count number of datapoints for this patient
    num_x = group_patient.shape[0]
    
    ## np.ones necessary to include constant in regression
    X = np.column_stack((x*x*x, x*x, x, np.ones(num_x)))

    ## OLS fit
    model = sm.OLS(y,X)
    results = model.fit()

    ## Option 1: Outliers defined to be either datapoints with large residuals
    infl = results.get_influence()
    resids = infl.resid_studentized_external
    ## Calculate residual cutoff
    resid_cutoff = stats.t.ppf(1.-alpha/2, results.df_resid)
    large_resid = np.abs(resids) > resid_cutoff

    ## Option 2: Outliers defined to be either datapoints with cooks distance greater than 4/N
    #(c, p) = infl.cooks_distance
    #cooks_cutoff = 4.0/group_patient.shape[0]
    #large_cooks = c > cooks_cutoff

    ## Option 3: .. or both
    #outliers_bool = np.logical_or(outliers_bool_norm_resid, outliers_bool_cooks_distance)

    ## Currently choose option 3 for defining outliers
    return large_resid

## Fit a OLS curve to the group patient's y as a function of x
## Has more restrictive critera than fit(), and therefore better to use when
## patient's measurements do not fluctuate as much
def fit2(group_patient, header_y, header_x):
    
    x = group_patient[header_x]
    y = group_patient[header_y]

    ## Count number of datapoints for this patient
    num_x = group_patient.shape[0]
    
    ## np.ones necessary to include constant in regression
    X = np.column_stack((np.log(x), np.ones(num_x)))
    
    ## OLS fit
    model = sm.OLS(y,X)
    fitted = model.fit()

    ## Print summar of fit
    #print fitted.summary()

    ## Option 1: Outliers defined to be either datapoints with residuals greater than 2.5 standard deviations
    outliers_bool_norm_resid = abs(fitted.norm_resid()) > 2.5

    ## Option 2: Outliers defined to be either datapoints with cooks distance greater than 4/(N-K-1)
    #influence = fitted.get_influence()
    #(c, p) = influence.cooks_distance
    #outliers_bool_cooks_distance = c > 4.0/(num_x-2)

    ## Option 3: .. or both
    #outliers_bool = np.logical_or(outliers_bool_norm_resid, outliers_bool_cooks_distance)

    ## Currently choose option 3 for defining outliers
    return outliers_bool_norm_resid

#####################

## Read in document
df = pd.read_csv(csv_location)

## Drop columns not needed
df = df.drop(["Height (ft&inc)","Body Mass Index Percentile","Age on 5/1/2012","BMI","Patient Ethnicity","Patient Race"], axis=1)
df.rename(columns={"Patient ID": "id", "Gender": "gender", "Race/Ethnicity": "race_ethnicity", "Patient Age At Encounter": "age", \
                   "Body Mass Index": "bmi", "Weight (Ounces)":"wt_ounces", "Height (Inches)":"ht"}, inplace=True)

## Convert numerical columns to floats (and remove #VALUE! if necessary)
cond = df["ht"] == "#VALUE!"
df["ht"][cond] = "NaN"
df = change_dtypes(df)

## Fill in missing values: need at least one of bmi, wt, ht to resolve third
selector = np.isnan(df["bmi"]) & ~np.isnan(df["wt_ounces"]) & ~np.isnan(df["ht"])
missing_series = df[selector]["wt_ounces"]/16/(np.power(df[selector]["ht"],2))*703
df["bmi"] = pd.concat([df["bmi"].dropna(), missing_series.dropna()]).reindex_like(df)

selector = ~np.isnan(df["bmi"]) & np.isnan(df["wt_ounces"]) & ~np.isnan(df["ht"])
missing_series = df[selector]["bmi"]*np.power(df[selector]["ht"],2)/703*16
df["wt"] = pd.concat([df["wt_ounces"].dropna(), missing_series.dropna()]).reindex_like(df)

selector = ~np.isnan(df["bmi"]) & ~np.isnan(df["wt_ounces"]) & np.isnan(df["ht"])
missing_series = np.power(df[selector]["wt_ounces"]/16/(df[selector]["bmi"])*703,0.5)
df["ht"] = pd.concat([df["ht"].dropna(), missing_series.dropna()]).reindex_like(df)

## Remove rows that can't be resolved for bmi, wt, or ht
selector = ~np.isnan(df["bmi"]) & ~np.isnan(df["wt_ounces"]) & ~np.isnan(df["ht"]) & df["gender"].notnull()
df = df[selector]

## Convert from ounces to pounds
df["wt"] = df["wt_ounces"]/16
df = df.drop("wt_ounces", axis=1)

## Remove bad patients
for patient_id in bad_patients:
    df[df["id"] == patient_id] = np.nan
df = df.dropna()

## Remove datapoints where ht = 0
df = df[df["ht"] != 0]
    
## Sort columns by header id and reindex
df = df.sort(["id", "age"], ascending=[1,1])
df = df.reindex()

## Group by patient id
grouped_patients = df.groupby("id")

df_resampled_list = []
df_filtered_list = []
df_filtered_out_list = []
df_outliers_list = []

## Iterate through each patient in the dataframe
## Note: using pandas to append is O(n^2), whereas appending list of dictionaries is O(n)
## Takes ~24 mins
for name_patient, group_patient in grouped_patients:
    ## To grab patients, type:
    #grouped_patients.get_group(np.int64(5035))

    ## Skip patients that have less than minimum number of datapoints
    if group_patient.shape[0] < min_datapoints:
        group_patient.apply(lambda x: df_filtered_out_list.append(x.to_dict()), axis = 1)
        #df_filtered_out = pd.DataFrame.append(df_filtered_out, group_patient)
        continue

    ## Change datapoints with age 0 to age 0.01 (in order to take its log for the regression)
    group_patient["age"][group_patient["age"] == np.float64(0)] = np.float64(0.01)

    ## Conduct regression of ht vs age and wt vs age to determine outliers
    
    ## Use fit2() if patient only has data from 15 yrs old and on, as
    ## patient's measurements do not change much
    if group_patient["age"].min() > 15:
        outliers_bool_ht = fit2(group_patient, "ht", "age")
        outliers_bool_wt = fit2(group_patient, "wt", "age")
    ## Otherwise use fit()
    else:
        outliers_bool_ht = fit(group_patient, "ht", "age")
        outliers_bool_wt = fit(group_patient, "wt", "age")
    outliers_bool = np.logical_or(outliers_bool_ht, outliers_bool_wt)

    ## Remove outlier datapoints and save removed datapoints to df_outliers
    if any(outliers_bool):
        group_patient[outliers_bool].apply(lambda x: df_outliers_list.append(x.to_dict()), axis = 1)
    group_patient = group_patient[np.logical_not(outliers_bool)]

    ## Check again after removing outliers
    ## Skip patients that have less than minimum number of datapoints
    if group_patient.shape[0] < min_datapoints:
        group_patient.apply(lambda x: df_filtered_out_list.append(x.to_dict()), axis = 1)
        continue
    
    ## Interpolate ht and wt linearly and resample at defined age intervals
##    f_ht = interpolate.interp1d(group_patient["age"], np.log(group_patient["ht"]), kind='cubic', bounds_error=False)
##    f_wt = interpolate.interp1d(group_patient["age"], np.log(group_patient["wt"]), kind='cubic', bounds_error=False)
##    df_resampled_patient = pd.DataFrame(intervals, columns=["age"])
##    df_resampled_patient["ht"] = np.power(math.e, df_resampled_patient["age"].apply(f_ht))
##    df_resampled_patient["wt"] = np.power(math.e, df_resampled_patient["age"].apply(f_wt))

    f_ht = interpolate.interp1d(group_patient["age"], group_patient["ht"], kind='linear', bounds_error=False)
    f_wt = interpolate.interp1d(group_patient["age"], group_patient["wt"], kind='linear', bounds_error=False)
    df_resampled_patient = pd.DataFrame(intervals, columns=["age"])
    df_resampled_patient["ht"] = df_resampled_patient["age"].apply(f_ht)
    df_resampled_patient["wt"] = df_resampled_patient["age"].apply(f_wt)

    ## Drop NAs, which occur because no extrapolation is conducted
    df_resampled_patient = df_resampled_patient.dropna()

    ## Recalculate BMIs
    df_resampled_patient["bmi"] = df_resampled_patient["wt"]/(np.power(df_resampled_patient["ht"],2))*703

    ## Save patient attributes
    df_resampled_patient["id"] = name_patient
    df_resampled_patient["gender"] = group_patient["gender"].max()
    df_resampled_patient["race_ethnicity"] = group_patient["race_ethnicity"].max()

    ## Save filtered and resampled data
    df_resampled_patient.apply(lambda x: df_resampled_list.append(x.to_dict()), axis = 1)
    group_patient.apply(lambda x: df_filtered_list.append(x.to_dict()), axis = 1)

## Convert list of dictionaries to dataframes
df_resampled = change_dtypes(pd.DataFrame(df_resampled_list).dropna())
df_filtered = change_dtypes(pd.DataFrame(df_filtered_list).dropna())
df_filtered_out = change_dtypes(pd.DataFrame(df_filtered_out_list).dropna())
df_outliers = change_dtypes(pd.DataFrame(df_outliers_list).dropna())

## Print dataframes to CSVs
df_resampled.to_csv("../../data/csv/BMI_resampled.csv", index_label=False, index=False)
df_filtered.to_csv("../../data/csv/BMI_filtered.csv", index_label=False, index=False)
df_filtered_out.to_csv("../../data/csv/BMI_filtered_out.csv", index_label=False, index=False)
df_outliers.to_csv("../../data/csv/BMI_outliers.csv", index_label=False, index=False)

## Dump files with pickle using the highest protocol available
output = open('../../data/pkl/BMI_resampled.pkl', 'wb')
pickle.dump(df_resampled, output, -1)
output.close()
output = open('../../data/pkl/BMI_filtered.pkl', 'wb')
pickle.dump(df_filtered, output, -1)
output.close()
output = open('../../data/pkl/BMI_filtered_out.pkl', 'wb')
pickle.dump(df_filtered_out, output, -1)
output.close()
output = open('../../data/pkl/BMI_outliers.pkl', 'wb')
pickle.dump(df_outliers, output, -1)
output.close()
