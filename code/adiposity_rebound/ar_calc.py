##### SETUP ######

import sys
sys.path.append('../config')
sys.path.append('../visualize')

import config
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import interpolate
import math

#import vis_quartiles_individual

##### VARIABLES ######

attributes = ["ht","wt","bmi"]
percentiles = config.percentiles
intervals = config.intervals

## Plot figures?
plot_bool = False

## Plot for blog style if plot_bool is true?
plot_for_blog = False

## Input individual patients? (Otherwise pulls all patients from BMI_filtered_contain_age5.pkl)
input_patient_bool = False

## Individual patient to analyze if input_patient_bool is true
patient_id = 10258 #16785, 10258, 9026, 12322

## Limit number of patients plot
count_bool = False

## Plot log on y axis?
plot_log = False
font_size = 14


######################

##### FUNCTIONS ######

def find_percentile(id_patient, age, grouped_age, grouped_patient_age):                                     
    df_datapoint = grouped_patient_age.get_group((id_patient, age))
    ht = df_datapoint["ht"]
    wt = df_datapoint["wt"]
    bmi = df_datapoint["bmi"]
    gender = df_datapoint["gender"]
                                     
    group_age = grouped_age.get_group(age)
    group_age = group_age[group_age["gender"] == gender.iloc[0]]
    
    num_patients = group_age.shape[0]
    percent_ht = np.sum(group_age["ht"] < ht.iloc[0])*100.0/num_patients
    percent_wt = np.sum(group_age["wt"] < wt.iloc[0])*100.0/num_patients
    percent_bmi = np.sum(group_age["bmi"] < bmi.iloc[0])*100.0/num_patients
    return (percent_ht, percent_wt, percent_bmi, ht.iloc[0], wt.iloc[0], bmi.iloc[0])

######################

## Open pickle file, saved from bmi_adiposity_rebound_filter.py
## df_range contains individuals that have data in the age range where AR occurs
df = pickle.load(open('../../data/pkl/BMI_filtered_contain_age5.pkl', 'rb'))

## Open pickle file, saved from bmi_initial_processing.py
df_resampled = pickle.load(open('../../data/pkl/BMI_resampled.pkl', 'rb'))

## Already ran calculation ##
## Keep only method 4 results ##
#pat_list = pd.read_csv("../../data/csv/BMI_ar_4_method4.csv", names = ["Patient ID"])
#df = df[df["Patient ID"].isin(pat_list["Patient ID"])]

## Group datapoints for each patient
grouped = df.groupby(["id"])

resampled_grouped_age = df_resampled.groupby(["age"])
resampled_grouped_patient_age = df_resampled.groupby(["id", "age"])

## Create new list to add AR information for each patient
df_ar_list = []
                           
## Iterate through each patient
count = 0
for name_patient, group_patient in grouped:

    #print name_patient
    if count_bool:
        count = count + 1
        if count < 0:
            continue
        if count == 20:
            break
    if input_patient_bool:
        name_patient = patient_id    #11995 #12552
        group_patient = grouped.get_group(name_patient)
    #print name_patient
    #print group_patient

    # Throw out data less than 2 yeras
    group_patient = group_patient[group_patient["age"] >= 2.0]
    
    if plot_bool and not plot_for_blog:
        ## Create figure to hold patient-specific plots 
        fig = plt.figure()
        ax_age_ht = fig.add_subplot(221)
        ax_age_wt = fig.add_subplot(222)
        ax_age_bmi = fig.add_subplot(223)
        ax_velo = fig.add_subplot(224) ## First derivative of log(ht), log(wt), and log(bmi)
    elif plot_for_blog:
        fig = plt.figure(1, figsize=(15, 4.75))
        fig.subplots_adjust(bottom=0.15, left=0.025, right=0.975)
        ax_age_ht = fig.add_subplot(131)
        ax_age_wt = fig.add_subplot(132) 
        ax_age_bmi = fig.add_subplot(133)
        fig2 = plt.figure()
        ax_velo = fig2.add_subplot(111)
        
    ## Count number of datapoints for this patient
    num_x = group_patient.shape[0]
    
    # And at least 5 datapoints?
    if num_x < 5:
        continue
    
    ## Regression of log(ht) on age
    x = group_patient["age"]
    log_ht = np.log(group_patient["ht"])
    y = log_ht

    ## Model with cubic equation
    X = np.column_stack((x*x*x, x*x, x, np.ones(num_x)))

    ## Make OLS model
    model = sm.OLS(y,X)
    results = model.fit()

    ## Sample fit
    b = results.params
    xnew = np.linspace(x.min(),x.max(),100)
    ynew_ht_log = b[0]*xnew*xnew*xnew + b[1]*xnew*xnew + b[2]*xnew + b[3]
    ynew_ht = np.power(math.e, ynew_ht_log)

    ## Calculate first derivative of log(ht)
    ynew_ht_log_velo = 3*b[0]*xnew*xnew + 2*b[1]*xnew + b[2]

    if plot_bool:
        ax_velo.plot(xnew,2* ynew_ht_log_velo, 'r-', label="2*d(log(Height))/dt")
        if plot_log:
            ## Plot original data points and fit on log-linear plot
            ax_age_ht.plot(x,y, 'ro')
            ax_age_ht.plot(xnew,ynew_ht_log, 'r-')
        else:
            ## Plot original data points and fit on lin-lin plot
            ax_age_ht.plot(x,group_patient["ht"], 'ro')
            ax_age_ht.plot(xnew,ynew_ht, 'r-')

    ## Regression of log(wt) on age
    x = group_patient["age"]
    log_wt = np.log(group_patient["wt"])
    y = log_wt

    ## Model with cubic equation
    X = np.column_stack((x*x*x, x*x, x, np.ones(num_x)))

    ## Make RLM model
    #model = sm.OLS(y,X)
    model = sm.RLM(y,X)

    ## Sample fit
    results = model.fit()
    b = results.params
    xnew = np.linspace(x.min(),x.max(),100)
    ynew_wt_log = b[0]*xnew*xnew*xnew + b[1]*xnew*xnew + b[2]*xnew + b[3]
    ynew_wt = np.power(math.e, ynew_wt_log)

    ## Calculate first derivative of log(ht)
    ynew_wt_log_velo = 3*b[0]*xnew*xnew + 2*b[1]*xnew + b[2]

    if plot_bool:
        ax_velo.plot(xnew,ynew_wt_log_velo, 'b-', label="d(log(Weight))/dt")
        if plot_log:
            ## Plot original data points and fit on lin-lin plot
            ax_age_wt.plot(x,y, 'bo')
            ax_age_wt.plot(xnew,ynew_wt_log, 'b-')
        else:
            ## Plot original data points and fit on log-linear plot
            ax_age_wt.plot(x,group_patient["wt"], 'bo')
            ax_age_wt.plot(xnew,ynew_wt, 'b-')

    ## Calculate first derivative of log(bmi) from first derivatives of log(wt) and log(ht)
    bmi_velo = np.subtract(ynew_wt_log_velo, 2*ynew_ht_log_velo)

    if plot_bool:
        ## Plot first derivative of log(bmi) on log-linear plot
        ax_velo.plot(xnew, bmi_velo, 'g-', label = "d(log(BMI))/dt")
        ax_velo.axhline(y=0)

        ## Plot original BMI datapoints
        ax_age_bmi.plot(group_patient["age"],group_patient["bmi"], 'go')

        ynew_bmi = np.divide(ynew_wt, np.power(ynew_ht, 2)) * 703
        ax_age_bmi.plot(xnew, ynew_bmi, 'g-')
   
        ## Set axis labels
        ax_age_ht.set_xlabel("Age (years)")
        ax_age_wt.set_xlabel("Age (years)")
        ax_age_bmi.set_xlabel("Age (years)")
        if plot_log:
            ax_age_ht.set_ylabel("log(Height (inches))")
            ax_age_wt.set_ylabel("log(Weight (pounds))")
        else:
            ax_age_ht.set_ylabel("Height (inches)")
            ax_age_wt.set_ylabel("Weight (pounds)")              
        ax_age_bmi.set_ylabel("BMI")

        ax_velo.set_xlabel("Age (years)")
        ax_velo.set_ylabel("d/dt(log(variable))")
        ax_velo.legend(loc = "lower right", prop={'size':font_size})

        ## Set title
        if not plot_for_blog:
            fig.suptitle("Patient #" + str(int(name_patient)) + " Progressing through Adiposity Rebound", fontsize = font_size)
        else:
            fig.suptitle("Individual Patient with Late Adiposity Rebound", fontsize = font_size)

        font = {'family' : 'normal', 'weight' : 'normal', 'size'   : font_size}
        plt.rc('font', **font)

    ## Create DataFrame with first derivatives
    df_derivs = pd.DataFrame({"age":xnew, "ht_velo": ynew_ht_log_velo, "wt_velo": ynew_wt_log_velo, "bmi_velo": bmi_velo})
    
    ## Adiposity rebound occurs when velocity of log(BMI) goes from negative to
    ## to positive. If the velocity stays positive, the ratio curve opens up,
    ## and thus the minimum of the curve is the adiposity rebound

    ## Default value of ar_age, indicating it could not be found
    ar_age = np.nan
    
    ## CASE 1: BMI derivative stays above 0. Pick the minimum.
    if df_derivs["bmi_velo"].min() > 0:
        ar_age = df_derivs["age"].ix[df_derivs["bmi_velo"].idxmin()]
        if plot_bool:
            ax_velo.axvline(x=ar_age)
        #print "here1"
        ar_find = 1

    ## CASE 2: BMI derivative stays below 0. Pick the maximum.
    elif df_derivs["bmi_velo"].max() < 0:
        ar_age = df_derivs["age"].ix[df_derivs["bmi_velo"].idxmax()]
        if plot_bool:
            ax_velo.axvline(x=ar_age)
        #print "here2"
        ar_find = 2
    
    ## Note that the BMI derivative might cross 0 twice
    else:
        ## Find the index of the datapoint with the largest BMI derivative
        max_bmi_velo_index = df_derivs["bmi_velo"].idxmax()
        min_bmi_velo_index = df_derivs["bmi_velo"].idxmin()

        ## Find the largest index
        largest_index = max(df_derivs["bmi_velo"].index)

        ## CASE 3: BMI curve starts > 0, ends < 0. Unusual because that means
        ## BMI goes through maximum. Maybe pick earliest age as AR?
        if df_derivs["bmi_velo"].iloc[0] > 0 and df_derivs["bmi_velo"].iloc[-1] < 0:
            ar_find = 3
            df_derivs_cut = df_derivs

        ## CASE 4: BMI curve starts < 0, ends > 0. 
        elif df_derivs["bmi_velo"].iloc[0] < 0 and df_derivs["bmi_velo"].iloc[-1] > 0:
            ar_find = 4
            df_derivs_cut = df_derivs.ix[min_bmi_velo_index:largest_index]
            
        ## CASE 5: BMI curve starts and ends > 0, and opens up, crossing 0 twice
        ## So extract portion from [minimum, right end] when BMI derivative crosses
        ## from negative to positive
        elif df_derivs["bmi_velo"].iloc[0] > 0 and df_derivs["bmi_velo"].iloc[-1] > 0:
            ar_find = 5
            df_derivs_cut = df_derivs.ix[min_bmi_velo_index:largest_index]

        ## CASE 6: BMI curve starts and ends < 0, and opens down, crossing 0 twice
        ## So extract portion from [left end, maximum] when BMI derivative crosses
        ## from negative to positive
        else:
            ar_find = 6
            df_derivs_cut = df_derivs.ix[0:max_bmi_velo_index]
            
        ## Linearly interpolate age on first derivative of log(bmi)
        #f = interpolate.InterpolatedUnivariateSpline(df_derivs_cut["bmi_velo"], df_derivs_cut["age"])
        f = interpolate.interp1d(df_derivs_cut["bmi_velo"], df_derivs_cut["age"])

        ## Try to find age at which first derivative of log(bmi) == 0
        try:
            ar_age = f(0)
            if plot_bool:
                ax_velo.axvline(x=ar_age)
                if plot_for_blog:
                    ax_age_bmi.axvline(x=ar_age, color = "red", ls = "--")

        ## If cannot interpolate, means that either no ages or two ages were found. Thus, cannot determine AR.
        except ValueError as inst:
            print "ERROR: " + str(name_patient) + " - " + inst.args[0]
        
    if plot_bool:
        ## Plot when patient's indvidual curve over population growth curves
        #vis_quartiles_individual.plot_individual_against_percentiles(name_patient)

        
        ## Display plot
        plt.show() #bbox_inches='tight', transparent="True", pad_inches=0

    ## Now calculate percentile patient is end at the end of his/her growth curve
            
    ## Use resampled dataset, and group by patients
    grouped_resampled_patient = df_resampled.groupby(["id"])
    grouped_resampled_patient_age = df_resampled.groupby(["id", "age"])

    ## Select the last age datapoint, and record age, height, weight, and BMI
    last_resampled_dpt = grouped_resampled_patient.get_group(name_patient).sort("age").iloc[-1]
    last_age = last_resampled_dpt["age"]
    last_ht = last_resampled_dpt["ht"]
    last_wt = last_resampled_dpt["wt"]
    last_bmi = last_resampled_dpt["bmi"]

    first_resampled_dpt = grouped_resampled_patient.get_group(name_patient).sort("age").iloc[0]
    first_age = first_resampled_dpt["age"]

    ## Calculate weight, height, BMI percentiles at AR, if AR was found

    if np.isnan(ar_age):
        ar_ht_perc = np.nan
        ar_wt_perc = np.nan
        ar_bmi_perc = np.nan

        ar_ht_res = np.nan
        ar_wt_res = np.nan
        ar_bmi_res = np.nan
        
    else:
        interval_index_right = np.searchsorted(config.intervals, ar_age)
        
        if ar_age in config.intervals:
            (ar_ht_perc, ar_wt_perc, ar_bmi_perc, ar_ht_res, ar_wt_res, ar_bmi_res) = find_percentile(name_patient, ar_age, resampled_grouped_age, resampled_grouped_patient_age)
        else:
            interval_index_left = interval_index_right - 1
            age_left = intervals[interval_index_left]
            age_right = intervals[interval_index_right]

            if age_left < group_patient["age"].min():
                (ar_ht_perc, ar_wt_perc, ar_bmi_perc, ar_ht_res, ar_wt_res, ar_bmi_res) = find_percentile(name_patient, age_right, resampled_grouped_age, resampled_grouped_patient_age)
                
            elif age_right > group_patient["age"].max():
                (ar_ht_perc, ar_wt_perc, ar_bmi_perc, ar_ht_res, ar_wt_res, ar_bmi_res) = find_percentile(name_patient, age_left, resampled_grouped_age, resampled_grouped_patient_age)
                
            else:
                (ht_left_perc, wt_left_perc, bmi_left_perc, ar_ht_left_res, ar_wt_left_res, ar_bmi_left_res) = \
                               find_percentile(name_patient, age_left, resampled_grouped_age, resampled_grouped_patient_age)
                (ht_right_perc, wt_right_perc, bmi_right_perc, ar_ht_right_res, ar_wt_right_res, ar_bmi_right_res) = \
                                find_percentile(name_patient, age_right, resampled_grouped_age, resampled_grouped_patient_age)
                    
                right_ratio = (ar_age - age_left)/(age_right - age_left)
                left_ratio = 1 - right_ratio
                ar_ht_perc = ht_left_perc * left_ratio + ht_right_perc * right_ratio
                ar_wt_perc = wt_left_perc * left_ratio + wt_right_perc * right_ratio
                ar_bmi_perc = bmi_left_perc * left_ratio + bmi_right_perc * right_ratio
                
                ar_ht_res = ar_ht_left_res * left_ratio + ar_ht_right_res * right_ratio
                ar_wt_res = ar_wt_left_res * left_ratio + ar_wt_right_res * right_ratio
                ar_bmi_res = ar_bmi_left_res * left_ratio + ar_bmi_right_res * right_ratio

    ## Create new numpy list that is of the correct dimensions
    #row = np.array([None]*len(header_list))
    #row = np.reshape(row, (1, len(header_list)))

    ## Create new DataFrame row
    #df_row = pd.DataFrame(row, columns = header_list)
    df_row = dict()

    ## Save patient information to this new row
    df_row["id"] = name_patient
    df_row["gender"] = group_patient["gender"].iloc[0]
    df_row["race_ethnicity"] = group_patient["race_ethnicity"].iloc[0]
    df_row["count"] = num_x
    df_row["ar_find"] = ar_find
    df_row["age_ar"] = ar_age
    df_row["age_front"] = first_age
    df_row["age_end"] = last_age

    ## Find and save height, weight, BMI percentiles at the last resampled datapoint
    (df_row["perc_ht_end"], df_row["perc_wt_end"], df_row["perc_bmi_end"], \
     df_row["res_ht_end"], df_row["res_wt_end"], df_row["res_bmi_end"]) = \
                            find_percentile(name_patient, last_age, resampled_grouped_age, resampled_grouped_patient_age)

    ## Find and save height, weight, BMI percentiles at the first resampled datapoint
    (df_row["perc_ht_front"], df_row["perc_wt_front"], df_row["perc_bmi_front"], \
     df_row["res_ht_front"], df_row["res_wt_front"], df_row["res_bmi_front"]) = \
                            find_percentile(name_patient, first_age, resampled_grouped_age, resampled_grouped_patient_age)

    ## Seight, weight, BMI percentiles at AR
    (df_row["perc_ht_ar"], df_row["perc_wt_ar"], df_row["perc_bmi_ar"]) = \
                            (ar_ht_perc, ar_wt_perc, ar_bmi_perc)
    (df_row["res_ht_ar"], df_row["res_wt_ar"], df_row["res_bmi_ar"]) = \
                            (ar_ht_res, ar_wt_res, ar_bmi_res)

    ## Append new row to aggregate AR info dataframe
    df_ar_list.append(df_row)
    #df_ar_info = pd.DataFrame.append(df_ar_info, df_row)

    if input_patient_bool:
        break
df_ar_info = pd.DataFrame(df_ar_list)
    
if not input_patient_bool:
    ## Dump data
    #output = open('../../data/pkl/BMI_ar_4_onlymethod4_calculate_5pt.pkl', 'wb')
    output = open('../../data/pkl/BMI_filtered_contain_age5_ar_calculate.pkl', 'wb')
    pickle.dump(df_ar_info, output, -1)
    output.close()

    #df_ar_info.to_csv("../../data/csv/BMI_ar_4_onlymethod4_calculate_5pt.csv")
    df_ar_info.to_csv("../../data/csv/BMI_filtered_contain_age5_ar_calculate.csv")
