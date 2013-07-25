##### SETUP ######

import sys
sys.path.append('../config')

import config
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

header = config.header
attributes = config.attributes
percentiles = config.percentiles

##################

### VARIABLES ###

## Age range to display on plot
x_age_min = 0
x_age_max = 20

#################

## Open pickle file, saved from bmi_aggregate.py
pkl_file = open('../../data/pkl/bmi_data_aggregate.pkl', 'rb')
df_aggregate = pickle.load(pkl_file)

       
## Create dictionary of race/ethnicity and gender to plotting characters
## For race/ethnicity, select legend label and line color
race_ethnicity_dict = {'Caucasian':('White','r'), \
            'African American':('Black','g'), \
            'Hispanic/Latino':('Hisp', 'b'), \
            #'Asian':('Asian','c'), \
            #'American Indian or Alaska Native':('Native Am.','m'), \
            #'Other':('Other','y')
            }
#color not used: k
## For gender, select line style
gender_dict = {'M':'-', 'F':'--'}

## Remove data that wouldn't be plotted (because outside interested age range)
df_aggregate[df_aggregate[header["age"]] < x_age_min] = np.nan
df_aggregate[df_aggregate[header["age"]] > x_age_max] = np.nan
df_aggregate = df_aggregate.dropna()

df_aggregate[df_aggregate[header["race_ethnicity"]] == 'American Indian or Alaska Native'] = np.nan
df_aggregate[df_aggregate[header["race_ethnicity"]] == 'Asian'] = np.nan
#df_aggregate[df_aggregate[header["race_ethnicity"]] == 'Hispanic/Latino'] = np.nan
df_aggregate[df_aggregate[header["race_ethnicity"]] == 'Other'] = np.nan
df_aggregate[df_aggregate[header["race_ethnicity"]] == 'All'] = np.nan
df_aggregate.dropna()

## Group aggregate information by gender and race/ethnicity
grouped = df_aggregate.groupby([header["gender"],header["race_ethnicity"]])

## Initialize dictionary to save figure handles
fig_dict = dict()

## Two figures: one for males, one for females
fig_list = ['M','F']

## For each figure,
for char in fig_list:
    ## Initialize dictionary to save axis handles
    fig_dict[char] = dict()

    ## Create figure
    fig_dict[char]['fig'] = plt.figure()

    ## Save axis handles
    ## Four suplots: age vs ht, avg vs wt, avg vs bmi, and ht vs wt
    fig_dict[char]['ax_ht'] = fig_dict[char]['fig'].add_subplot(221)
    fig_dict[char]['ax_wt'] = fig_dict[char]['fig'].add_subplot(222)
    fig_dict[char]['ax_bmi'] = fig_dict[char]['fig'].add_subplot(223)
    #fig_dict[char]['ax_ht_wt'] = fig_dict[char]['fig'].add_subplot(224)

## Iterate through all gender and race/ethnicity groups
for name, group in grouped:
    name_gender = name[0]
    name_race_ethnicity = name[1]

    ## Lookup legend label for this group
    name_race_ethnicity_short = race_ethnicity_dict[name_race_ethnicity][0]
    cat_label = name_gender + ", " + name_race_ethnicity_short
    
    ## Look up line color and style for this group
    cat_color = race_ethnicity_dict[name_race_ethnicity][1]
    cat_line = gender_dict[name_gender]
    
    ## Plot group trends on each of the four subplots
    
    alpha_val = 0.1
    
    fig_dict[name_gender]['ax_ht'].fill_between(group[header["age"]].tolist(), group[header["ht_25"]].tolist(), group[header["ht_75"]].tolist(), facecolor=cat_color, alpha=alpha_val)

    line_ht = group.plot(x=header["age"], y=header["ht_50"], ax=fig_dict[name_gender]['ax_ht'], color=cat_color, linestyle=cat_line, label=cat_label)

    fig_dict[name_gender]['ax_wt'].fill_between(group[header["age"]].tolist(), group[header["wt_25"]].tolist(), group[header["wt_75"]].tolist(), facecolor=cat_color, alpha=alpha_val)

    line_wt = group.plot(x=header["age"], y=header["wt_50"], ax=fig_dict[name_gender]['ax_wt'], color=cat_color, linestyle=cat_line, label=cat_label)

    fig_dict[name_gender]['ax_bmi'].fill_between(group[header["age"]].tolist(), group[header["bmi_25"]].tolist(), group[header["bmi_75"]].tolist(), facecolor=cat_color, alpha=alpha_val)

    line_bmi = group.plot(x=header["age"], y=header["bmi_50"], ax=fig_dict[name_gender]['ax_bmi'], color=cat_color, linestyle=cat_line, label=cat_label)

    #line_ht_wt = group.plot(x=header["ht_50"], y=header["wt_50"], ax=fig_dict[name_gender]['ax_ht_wt'], color=cat_color, linestyle=cat_line, label=cat_label)

## For each figure, 
for char in fig_list:
    ## Set correct y labels
    fig_dict[char]['ax_ht'].set_ylabel("Height (inches)")
    fig_dict[char]['ax_wt'].set_ylabel("Weight (pounds)")
    fig_dict[char]['ax_bmi'].set_ylabel("BMI")
    #fig_dict[char]['ax_ht_wt'].set_ylabel("Weight (ounces)")

    fig_dict[char]['ax_ht'].set_xlabel("Age (years)")
    fig_dict[char]['ax_wt'].set_xlabel("Age (years)")
    fig_dict[char]['ax_bmi'].set_xlabel("Age (years)")
    #fig_dict[char]['ax_ht_wt'].set_xlabel("Height")

    fig_dict[char]['ax_ht'].set_xlim([x_age_min, x_age_max])
    fig_dict[char]['ax_wt'].set_xlim([x_age_min, x_age_max])
    fig_dict[char]['ax_bmi'].set_xlim([x_age_min, x_age_max])

    #handles, labels = ax_ht.get_legend_handles_labels()
    #labels = labels_list
    #ax_ht.legend(handles, labels)

    ## Place figure legend - in progress
    fig_dict[char]['ax_bmi'].legend(bbox_to_anchor=(1.2, 0.5), loc='center left', borderaxespad=0.)

    ## Set figure title
    fig_dict[char]['fig'].suptitle('Patient Statistics, ' + char, fontsize=12)

## Show final figures
plt.show()
