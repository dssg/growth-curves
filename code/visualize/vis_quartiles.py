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
header_list = config.header_list

attributes = config.attributes

intervals = config.intervals

percentiles = np.array([.50, .85]) #bmi_funcs.percentiles

###################

### VARIABLES ###

## Age range to display on plot
x_age_min = 0
x_age_max = 10

#################

## Open pickle file, saved from bmi_aggregate.py
pkl_file = open('../../data/pkl/bmi_data_aggregate.pkl', 'rb')
df_aggregate = pickle.load(pkl_file)

## Create dictionary of race/ethnicity and gender to plotting characters
## For race/ethnicity, select legend label and line color
race_ethnicity_dict = {'Caucasian':('White','r'), \
            'African American':('Black','g'), \
            'Hispanic/Latino':('Hisp', 'b'), \
            'Asian':('Asian','c'), \
            'American Indian or Alaska Native':('Native Am.','m'), \
            'Other':('Other','y')}
#color not used: k
## For gender, select line style
gender_dict = {'M':'-', 'F':'--'}

## Remove data that wouldn't be plotted (because outside interested age range)
df_aggregate[df_aggregate[header["age"]] < x_age_min] = np.nan
df_aggregate[df_aggregate[header["age"]] > x_age_max] = np.nan

df_aggregate[df_aggregate[header["race_ethnicity"]] != 'All'] = np.nan
df_aggregate = df_aggregate.dropna()

#### CDC ADD ####
df_cdc = pd.read_csv("../../data/csv/cdc_data.csv")
df_cdc = df_cdc[df_cdc[header["age"]] < 19.5]

df_cdc = df_cdc[df_cdc[header["age"]] > x_age_min]
df_cdc = df_cdc[df_cdc[header["age"]] < x_age_max]
#################

## Group aggregate information by gender and race/ethnicity
#grouped = df_aggregate.groupby([header["gender"],header["race_ethnicity"]])
grouped = df_aggregate.groupby([header["gender"]])

#### CDC ADD ####
grouped_cdc = df_cdc.groupby([header["gender"]])
#################

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
    fig_dict[char]['ax_age_ht'] = fig_dict[char]['fig'].add_subplot(221)
    fig_dict[char]['ax_age_wt'] = fig_dict[char]['fig'].add_subplot(222)
    fig_dict[char]['ax_age_bmi'] = fig_dict[char]['fig'].add_subplot(223)
    #fig_dict[char]['ax_ht_wt'] = fig_dict[char]['fig'].add_subplot(224)f

percentiles_int, num_percentiles = config.convert_percentiles(percentiles)

## Iterate through all gender and race/ethnicity groups
for name, group in grouped:
    name_gender = name
    #name_race_ethnicity = name[1]
    #name_race_ethnicity_short = race_ethnicity_dict[name_race_ethnicity][0]

    
    
    ## Plot group trends on each of the four subplots
    attribute_pairs = [("age","ht"),("age","wt"),("age","bmi")]
    for x_attribute, y_attribute in attribute_pairs:
        color = config.get_color(len(percentiles))
        for percentile in percentiles_int:
            
            x_name = header[x_attribute]
            y_name = header[y_attribute + "_" + str(percentile)]
            ax_name = 'ax_'+x_attribute+'_'+y_attribute
            cat_label = str(percentile) + "%"
            cat_linewidth = 1.0
            cat_color = next(color)
            if percentile == 85:
                cat_linewidth = 3.0
                cat_label = cat_label + ", Childhood Overweight"
                cat_85_color = cat_color
            elif percentile == 95:
                cat_linewidth = 3.0
                cat_label = cat_label + ", Childhood Obese"
                cat_95_color = cat_color

            line = group.plot(x_name, y_name, ax=fig_dict[name_gender][ax_name], color=cat_color, label=cat_label, linewidth=cat_linewidth, ls="-")

            #### CDC ADD ####
            if percentile == 95:
                line = grouped_cdc.get_group(name_gender).plot(x_name, y_name, ax=fig_dict[name_gender][ax_name], color=cat_color, linewidth=3.0, ls="--")
            #################
       
## For each figure, 
for char in ['M','F']:
    ## Set correct y labels
    fig_dict[char]['ax_age_ht'].set_ylabel("Height (inches)")
    fig_dict[char]['ax_age_wt'].set_ylabel("Weight (pounds)")
    fig_dict[char]['ax_age_bmi'].set_ylabel("BMI")

    #fig_dict[char]['ax_age_ht'].set_xlim([x_age_min, x_age_max])
    #fig_dict[char]['ax_age_wt'].set_xlim([x_age_min, x_age_max])
    #fig_dict[char]['ax_age_bmi'].set_xlim([x_age_min, x_age_max])

    #fig_dict[char]['ax_age_bmi'].axhline(y=18.5, linewidth=4, color='r')
    #fig_dict[char]['ax_age_bmi'].axhline(y=25, linewidth=4, color='r')
    #fig_dict[char]['ax_age_bmi'].axhline(y=30, linewidth=4, color='r')

    #fig_dict[char]['ax_age_bmi'].plot([19.25, 20], [25, 25], linewidth=4, color=cat_85_color, ls=":", label="Adult Overweight (by definition)")
    #fig_dict[char]['ax_age_bmi'].plot([19.25, 20], [30, 30], linewidth=4, color=cat_95_color, ls=":", label="Adult Obese (by definition)")

    #print cdc data
##    for x_attribute, y_attribute in attribute_pairs:
##        percentile = 95
##        x_name = header[x_attribute]
##        y_name = header[y_attribute + "_" + str(percentile)]
##        ax_name = 'ax_'+x_attribute+'_'+y_attribute
##        cat_label = str(percentile) + "%"
##        line = grouped_cdc.get_group(char).plot(x_name, y_name, ax=fig_dict[char][ax_name], color=cat_95_color, linewidth=3.0, ls="--", \
##                                                       label="95%, CDC Growth Curves")
                
    #fig_dict[char]['ax_age_bmi'].plot([19.25, 19.25], [25, 30], 'D')
    
    handles, labels = fig_dict[char]['ax_age_bmi'].get_legend_handles_labels()

    h1 = zip(handles, labels)
    h1 = filter(lambda (x, y): y != header["age"], h1)

    handles, labels = zip(*h1)
    handles = list(handles)
    labels = list(labels)
    
    ## Invert legend order
    
    ## Place legend
    #fig_dict[char]['ax_age_bmi'].legend(handles[:-3][::-1] + [handles[-1]] + [handles[-2]] + [handles[-3]], labels[:-3][::-1] + [labels[-1]] + [labels[-2]] + [labels[-3]], bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)

    #legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)

    ## Set figure title
    fig_dict[char]['fig'].patch.set_facecolor('white')
    fig_dict[char]['fig'].suptitle('Patient Statistics, ' + char, fontsize=12)
    fig_dict[char]['fig'].savefig('caucasian_quartiles_'+char+'.png', facecolor=fig_dict[char]['fig'].get_facecolor(), edgecolor='none')

    fig_dict[char]['ax_age_ht'].set_xlabel("Age (years)")
    fig_dict[char]['ax_age_wt'].set_xlabel("Age (years)")
    fig_dict[char]['ax_age_bmi'].set_xlabel("Age (years)")
    
## Show final figures
plt.show()
