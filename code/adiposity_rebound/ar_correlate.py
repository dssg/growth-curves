##### SETUP ######

import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import interpolate

font_size = 14

#################################################################################

## Load files

## Individuals containing the age of 5
df = pickle.load(open('../../data/pkl/BMI_filtered_contain_age5_ar_calculate.pkl', 'rb'))
## All individuals, but at resampled intervals
df_resampled = pickle.load(open('../../data/pkl/BMI_resampled.pkl', 'rb'))

#################################################################################

## Export table of percentages of each type of adiposity rebound calculation

grouped_ar_find = df.groupby("ar_find")
counts = grouped_ar_find["id"].count()
percentages = grouped_ar_find["id"].count()*100.0/df.shape[0]

bmi_deriv_desc = ["BMI deriv > 0", \
"BMI deriv < 0", \
"BMI deriv starts > 0, ends < 0", \
"BMI deriv starts < 0, ends > 0", \
"BMI deriv starts > 0, dips < 0, ends > 0", \
"BMI deriv starts < 0, dips > 0, ends < 0"]

bmi_desc = ["BMI constantly increases", \
"BMI constantly decreases", \
"BMI passes through max", \
"BMI passes through min", \
"BMI passes through local max then local min", \
"BMI passes through local min then local max"]

res_summary = pd.DataFrame({"Counts":counts, "Percentages":percentages, \
                    "BMI Deriv":bmi_deriv_desc, "BMI":bmi_desc})
res_summary.to_csv("../../data/csv/BMI_ar_calculations_summary.csv")

#################################################################################

## Select only "good cases" to look at

## Case 4 is when BMI curve starts < 0, ends > 0.
good_cases = [4]
agg_bool = df.shape[0]*[False]
for case in good_cases:
    agg_bool = np.logical_or(agg_bool, df["ar_find"] == case)
df = df[agg_bool]

## Limit analysis to cases when adiposity rebound age is greater than 2
df = df[df["age_ar"] >= 2]

## Save individuals that satisfy these conditions
df.to_csv("../../data/csv/BMI_filtered_contain_age5_ar_calculate_method4.csv", index = False, index_label = False)

## Count the number of individuals in our dataset
num_x = df.shape[0]

## Possible dependent variables
x1 = df["age_ar"]
x2 = df["perc_bmi_ar"]
x3 = df["res_bmi_ar"]
x4 = df["age_end"]
x5 = df["perc_ht_ar"]
x6 = df["perc_wt_ar"]
x7 = df["perc_bmi_front"]

## Baseline of Caucasian Male
## Possible races/ethnicities: 'Caucasian', 'African American', 'Other', 'Asian', 'American Indian or Alaska Native', 'Hispanic/Latino'
re1 = df["race_ethnicity"] == 'African American'
re2 = df["race_ethnicity"] == 'Other'
re3 = df["race_ethnicity"] == 'Asian'
re4 = df["race_ethnicity"] == 'American Indian or Alaska Native'
re5 = df["race_ethnicity"] == 'Hispanic/Latino'
gen = df["gender"] == 'F'

## Code to shuffle adiposity rebound ages
#import random
#age_ar_list = df["age_ar"].tolist()
#random.shuffle(age_ar_list)
#x1 = np.array(age_ar_list)

## Regression on just age at adiposity rebound
X = np.column_stack((x1, np.ones(num_x)))

## Possible other choices for dependent variables
#X = np.column_stack((x1, x2, np.ones(num_x)))
#X = np.column_stack((x1, x2, gen, re1, re2, re3, re4, re5, gen*re1, gen*re2, gen*re3, gen*re4, gen*re5, np.ones(num_x)))
#X = np.column_stack((x1, gen, re1, re2, re3, re4, re5, gen*re1, gen*re2, gen*re3, gen*re4, gen*re5, np.ones(num_x)))

## Regression of BMI percentile at end of each growth curve
y = df["perc_bmi_end"]

## Could seek to do regression of resampled BMI at end of each growth curve
#y = df["res_bmi_end"]

## Create OLS model
model = sm.OLS(y,X)
results = model.fit()
b = results.params
xnew = np.linspace(x1.min(),x1.max(),100)
ynew = b[0]*xnew + b[1]

#################################################################################

## Plot BMI percentile at end of growth curve against age at adiposity rebound
fig = plt.figure()
ax = fig.add_subplot(111)
fig.suptitle("Regression on Individuals with Measureable Adiposity Rebound", fontsize = font_size)
ax.plot(x1, y, 'ro')
ax.set_xlabel("Age at Individual's Adiposity Rebound (years)")
ax.set_ylabel("BMI Percentile at End of Individual's Growth Curve")

## Plot regression line
ax.plot(xnew,ynew, 'g-', label = "Regression", linewidth=2.0)

## Set dashed lines to mark overweight and obese categories
ax.axhline(y=85, ls = "--", label = "Overweight", color= "blue", linewidth=2.0)
ax.axhline(y=95, ls = "--", label = "Obese", color = "purple", linewidth=2.0)

## Shrink plot to the left to make space for the legend
ax.set_position([0.1,0.1,.6,0.8])

## Plot the legend
ax.legend(bbox_to_anchor=(1.45, 1.02))

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(font_size)
    
## Show the plot
plt.show()

## Print the regression results
print results.summary()

#################################################################################

## Plot final BMI percentile on initial BMI percentile

## New figure
fig = plt.figure()

x = x7
X = np.column_stack((x, np.ones(num_x)))
model = sm.OLS(y,X)
results = model.fit()
b = results.params
xnew = np.linspace(x.min(),x.max(),100)
ynew = b[0]*xnew + b[1]

ax = fig.add_subplot(121)
ax.set_title("Final BMI vs. Initial BMI", fontsize=font_size)
ax.plot(x, y, 'ro')
ax.set_xlabel("BMI Percentile at Beginning of Individual's Growth Curve")
ax.set_ylabel("BMI Percentile at End of Individual's Growth Curve")
ax.plot(xnew,ynew, 'g-', label = "Regression", linewidth=2.0)
ax.axhline(y=85, ls = "--", label = "Overweight", color= "blue", linewidth=2.0)
ax.axhline(y=95, ls = "--", label = "Obese", color = "purple", linewidth=2.0)
ax.axvline(x=85, ls = "--", color= "blue", linewidth=2.0)
ax.axvline(x=95, ls = "--", color= "purple", linewidth=2.0)
ax.plot(np.linspace(0,100,100), np.linspace(0,100,100), ls = ":", color = "black")
ax.set_xlim([0,100])
ax.set_ylim([0,100])
ax.set_position([0.1,0.1,0.32,0.8])

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(font_size)

x = x2
X = np.column_stack((x, np.ones(num_x)))
model = sm.OLS(y,X)
results = model.fit()
b = results.params
xnew = np.linspace(x.min(),x.max(),100)
ynew = b[0]*xnew + b[1]

ax = fig.add_subplot(122)
ax.set_title("Final BMI vs. Adiposity Rebound BMI",fontsize=font_size)
ax.plot(x,y, 'ro')
ax.set_xlabel("BMI Percentile at Adiposity Rebound of Individual's Growth Curve")
ax.set_ylabel("BMI Percentile at End of Individual's Growth Curve")
ax.plot(xnew,ynew, 'g-', label = "Regression", linewidth=2.0)
ax.axhline(y=85, ls = "--", label = "Overweight", color= "blue", linewidth=2.0)
ax.axhline(y=95, ls = "--", label = "Obese", color = "purple", linewidth=2.0)
ax.axvline(x=85, ls = "--", color= "blue", linewidth=2.0)
ax.axvline(x=95, ls = "--", color= "purple", linewidth=2.0)
ax.plot(np.linspace(0,100,100), np.linspace(0,100,100), ls = ":", color = "black")
ax.set_xlim([0,100])
ax.set_ylim([0,100])

ax.set_position([0.5,0.1,0.32,0.8])
ax.legend(bbox_to_anchor=(1.45, 1.02))

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(font_size)
        
plt.show()

fig = plt.figure()
fig.subplots_adjust(bottom=0.15)
ax1 = fig.add_subplot(111)
ax1.hist(x1, bins = range(2,10))
ax1.set_xlabel("Age at Adiposity Rebound (year)")
ax1.set_ylabel("Count")
ax1.set_title("Histogram of When Adipsoity Rebound Occurs")
for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] + ax1.get_xticklabels() + ax1.get_yticklabels()):
    item.set_fontsize(font_size)
plt.show()

#################################################################################

## Plot additional figures

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.hist(x1)
ax1.set_xlabel("Age at Adiposity Rebound (year)")
ax1.set_ylabel("Count")

ax2 = fig.add_subplot(222)
ax2.hist(x2)
ax2.set_xlabel("BMI Percentile % at Adiposity Rebound")
ax2.set_ylabel("Count")

ax4 = fig.add_subplot(223)
ax4.hist(df["age_end"])
ax4.set_xlabel("Age at end of curve (year)")
ax4.set_ylabel("Count")

ax3 = fig.add_subplot(224)
ax3.hist(y)
ax3.set_xlabel("BMI Percentile % at End of Individual's Growth Curve")
ax3.set_ylabel("Count")
plt.show()

df["age_range_end_front"] = df["age_end"] - df["age_front"]
df["age_range_end_ar"] = df["age_end"] - df["age_ar"]
#plt.hist(df["age_range_end_front"])
#plt.show()
#plt.hist(df["age_range_end_ar"])
#plt.show()

#################################################################################

## Determine if high correlation of BMI percentile at adiposity rebound with BMI percentile
## at end of growth curve is due to the fact that limited time occurs between these two points
## (as opposed to the BMI percentile at adiposity rebound actually having significance)

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

def find_percentile_wrapper(name_patient, age, resampled_grouped_age, resampled_grouped_patient_age, grouped):
    group_patient = grouped.get_group(name_patient)
    interval_index_right = np.searchsorted(config.intervals, age)
    intervals = config.intervals.tolist()
    if interval_index_right == 0:
        interval_index_left = 0
    else:
        interval_index_left = interval_index_right - 1
    age_left = intervals[interval_index_left]
    age_right = intervals[interval_index_right]

    if age_left <= group_patient["age"].min():
        (ar_ht_perc, ar_wt_perc, ar_bmi_perc, ar_ht_res, ar_wt_res, ar_bmi_res) = find_percentile(name_patient, group_patient["age"].min(), resampled_grouped_age, resampled_grouped_patient_age)
        
    elif age_right >= group_patient["age"].max():
        (ar_ht_perc, ar_wt_perc, ar_bmi_perc, ar_ht_res, ar_wt_res, ar_bmi_res) = find_percentile(name_patient, group_patient["age"].max(), resampled_grouped_age, resampled_grouped_patient_age)
        
    else:
        (ht_left_perc, wt_left_perc, bmi_left_perc, ar_ht_left_res, ar_wt_left_res, ar_bmi_left_res) = \
                       find_percentile(name_patient, age_left, resampled_grouped_age, resampled_grouped_patient_age)
        (ht_right_perc, wt_right_perc, bmi_right_perc, ar_ht_right_res, ar_wt_right_res, ar_bmi_right_res) = \
                        find_percentile(name_patient, age_right, resampled_grouped_age, resampled_grouped_patient_age)
            
        right_ratio = (age - age_left)/(age_right - age_left)
        left_ratio = 1 - right_ratio
        ar_ht_perc = ht_left_perc * left_ratio + ht_right_perc * right_ratio
        ar_wt_perc = wt_left_perc * left_ratio + wt_right_perc * right_ratio
        ar_bmi_perc = bmi_left_perc * left_ratio + bmi_right_perc * right_ratio
        
        ar_ht_res = ar_ht_left_res * left_ratio + ar_ht_right_res * right_ratio
        ar_wt_res = ar_wt_left_res * left_ratio + ar_wt_right_res * right_ratio
        ar_bmi_res = ar_bmi_left_res * left_ratio + ar_bmi_right_res * right_ratio
    return ar_bmi_perc

## Set index of dataframe to the patient id
df = df.set_index("id")

## Shuffle the age difference between that at adiposity rebound and that at end of growth curve
import random
age_diff_list = df["age_range_end_ar"].tolist()
random.shuffle(age_diff_list)
df["age_diff_random"] = age_diff_list

## Select random age for each individual
df["age_random"] = df["age_end"] - df["age_diff_random"]

resampled_grouped_age = df_resampled.groupby(["age"])
resampled_grouped_patient = df_resampled.groupby(["id"])
resampled_grouped_patient_age = df_resampled.groupby(["id", "age"])
grouped = df_resampled.groupby(["id"])

## Calculate BMI percentile at this random age
df["perc_bmi_random"] = df.index.map(lambda name_patient: find_percentile_wrapper(name_patient, df.loc[name_patient, "age_random"], resampled_grouped_age, resampled_grouped_patient_age, grouped))

## Conduct OLS regression of BMI percentile at end of growth curve against BMI percentile at this random age
x = df["perc_bmi_random"]
X = np.column_stack((x, np.ones(num_x)))
model = sm.OLS(y,X)
results = model.fit()
b = results.params
xnew = np.linspace(x.min(),x.max(),100)
ynew = b[0]*xnew + b[1]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Final BMI vs. BMI at Random Age", fontsize=font_size)
ax.plot(x, y, 'ro')
ax.set_xlabel("BMI Percentile at Random Point along Growth Curve")
ax.set_ylabel("BMI Percentile at End of Growth Curve")
ax.plot(xnew,ynew, 'g-', label = "Regression", linewidth=2.0)
ax.axhline(y=85, ls = "--", label = "Overweight", color= "blue", linewidth=2.0)
ax.axhline(y=95, ls = "--", label = "Obese", color = "purple", linewidth=2.0)
ax.axvline(x=85, ls = "--", color= "blue", linewidth=2.0)
ax.axvline(x=95, ls = "--", color= "purple", linewidth=2.0)
ax.plot(np.linspace(0,100,100), np.linspace(0,100,100), ls = ":", color = "black")
ax.set_xlim([0,100])
ax.set_ylim([0,100])
plt.show()
print results.summary()
