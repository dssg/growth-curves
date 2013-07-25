##### SETUP ######

import sys
sys.path.append('../config')

import config
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

percentiles = np.array([.03, .05, .1, .25, .50, .75, .85, .90, .95, .97])
percentiles_expansion = np.array([.05, .33, .66, .80])
percentiles_dict = {85:5, 90:33, 95:66, 97:80}

percentiles_int, num_percentiles = config.convert_percentiles(percentiles)
percentiles_expansion_int, num_expansion_percentiles = config.convert_percentiles(percentiles_expansion)

header = config.header
header_list = config.header_list

attributes = {"bmi":"BMI"}

header, header_list = config.add_attributes_to_header_dict(attributes, header, header_list, percentiles_expansion)

###################

### VARIABLES ###

## Age range to display on plot
x_age_min = 0
x_age_max = 19.5

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

df_cdc = df_cdc[df_cdc[header["age"]] < x_age_min]
df_cdc = df_cdc[df_cdc[header["age"]] > x_age_max]
#################

#### EXPANSION ADD ####
pkl_file = open('../../data/pkl/bmi_data_expansion_aggregate.pkl', 'rb')
df_cdc = pickle.load(pkl_file)

#######################

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
    fig_dict[char]['ax_age_bmi'] = fig_dict[char]['fig'].add_subplot(111)

## Iterate through all gender and race/ethnicity groups
for name, group in grouped:
    name_gender = name
    #name_race_ethnicity = name[1]
    #name_race_ethnicity_short = race_ethnicity_dict[name_race_ethnicity][0]

    ## Plot group trends on each of the four subplots
    attribute_pairs = [("age","bmi")]
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

            line = group.plot(x_name, y_name, ax=fig_dict[name_gender][ax_name], color=cat_color, label=cat_label, linewidth=cat_linewidth, ls="--", alpha=0.5)

            ##### CDC ADD ####
            #if percentile != 85:
            #    line = grouped_cdc.get_group(name_gender).plot(x_name, y_name, ax=fig_dict[name_gender][ax_name], color=cat_color, linewidth=cat_linewidth/2, ls="--")
            ##################

            #### EXPANSION ADD ####

            if percentile in percentiles_dict.keys():
                percentile = percentiles_dict[percentile]
                y_name = header[y_attribute + "_" + str(percentile)]
                line = grouped_cdc.get_group(name_gender).plot(x_name, y_name, ax=fig_dict[name_gender][ax_name], color=cat_color, linewidth=3.0, ls="-")

            #######################
                
        
       
## For each figure, 
for char in fig_list:
    ## Set correct y labels
    fig_dict[char]['ax_age_bmi'].set_ylabel("BMI")

    fig_dict[char]['ax_age_bmi'].set_xlabel("Age (years)")

    #fig_dict[char]['ax_ht'].set_xlim([x_age_min, x_age_max])
    #fig_dict[char]['ax_wt'].set_xlim([x_age_min, x_age_max])
    #fig_dict[char]['ax_bmi'].set_xlim([x_age_min, x_age_max])

    #handles, labels = ax_ht.get_legend_handles_labels()
    #labels = labels_list
    #ax_ht.legend(handles, labels)


    #fig_dict[char]['ax_age_bmi'].axhline(y=18.5, linewidth=4, color='r')
    #fig_dict[char]['ax_age_bmi'].axhline(y=25, linewidth=4, color='r')
    #fig_dict[char]['ax_age_bmi'].axhline(y=30, linewidth=4, color='r')
    fig_dict[char]['ax_age_bmi'].plot([19.25, 20], [25, 25], linewidth=4, color=cat_85_color, ls="--", label="Adult Overweight (by definition)")
    fig_dict[char]['ax_age_bmi'].plot([19.25, 20], [30, 30], linewidth=4, color=cat_95_color, ls="--", label="Adult Obese (by definition)")

    fig_dict[char]['ax_age_bmi'].plot([19.25, 19.25], [25, 30], 'D')
    #fig_dict[char]['ax_age_bmi'].plot(19, 30, 'D')
    
    handles, labels = fig_dict[char]['ax_age_bmi'].get_legend_handles_labels()

    h1 = zip(handles, labels)
    h1 = filter(lambda (x, y): y != header["age"], h1)

    handles, labels = zip(*h1)
    handles = list(handles)
    labels = list(labels)
    
    ## Invert legend order
    
    ## Place legend
    box = fig_dict[char]['ax_age_bmi'].get_position()
    fig_dict[char]['ax_age_bmi'].set_position([box.x0, box.y0, box.width * 0.75, box.height])

    fig_dict[char]['ax_age_bmi'].legend(handles[:-2][::-1] + [handles[-1]] + [handles[-2]], labels[:-2][::-1] + [labels[-1]] + [labels[-2]], bbox_to_anchor=(1.025, 0.5), loc='center left', borderaxespad=0.)

    #legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)

    cdc_group = grouped_cdc.get_group(char)
    fig_dict[char][ax_name].fill_between(cdc_group[header["age"]].tolist(), cdc_group[header["bmi_5"]].tolist(), cdc_group[header["bmi_80"]].tolist(), facecolor='yellow', alpha=0.3)
    
    ## Set figure title
    fig_dict[char]['fig'].patch.set_facecolor('white')
    fig_dict[char]['fig'].suptitle('Patient Statistics, ' + char, fontsize=12)
    fig_dict[char]['fig'].savefig('caucasian_quartiles_'+char+'.png', facecolor=fig_dict[char]['fig'].get_facecolor(), edgecolor='none')

    fig_dict[char]['fig']
## Show final figures
plt.show()
