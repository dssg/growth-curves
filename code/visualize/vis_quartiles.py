## Draws growth curves from Northshore and CDC data

##### SETUP ######

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import matplotlib

###################

### VARIABLES ###

## Age intervals
## Float instability created errors, so corrected the following trunction 
intervals = np.trunc(np.concatenate((np.array([0.01]),np.arange(.05,.2,.05),np.arange(.4,2,.2), np.arange(2,20,.5)))*100)/100


## Percentiles to display
percentiles = np.array([10, 50, 75, 85, 90, 95, 97])

## Age range to display on plot
x_age_min = 2
x_age_max = 18.5

## Set to True to plot age vs. bmi, False to plot age vs. ht and age vs. wt
plot_bmi_only = True

## Display CDC data on plots
display_cdc = True

## Plot text size
font_size = 14

#################

## Color generator with maximum spacing between colors
## from: http://stackoverflow.com/questions/10254207/color-and-line-writing-using-matplotlib
import colorsys

def get_color(color):
    for hue in range(color):
        hue = 1. * hue / color
        col = [int(x) for x in colorsys.hsv_to_rgb(hue, 1.0, 230)]
        yield "#{0:02x}{1:02x}{2:02x}".format(*col)

#################

## Open pickle file, saved from bmi_aggregate.py
df_aggregate = pickle.load(open('../../data/pkl/BMI_aggregate_percentiles.pkl', 'rb'))

## Create dictionary of race/ethnicity and gender to plotting characters
## For race/ethnicity, select legend label and line color
#race_ethnicity_dict = {'Caucasian':('White','r'), \
#            'African American':('Black','g'), \
#            'Hispanic/Latino':('Hisp', 'b'), \
#            'Asian':('Asian','c'), \
#            'American Indian or Alaska Native':('Native Am.','m'), \
#            'Other':('Other','y')}
#color not used: k

## Remove data that wouldn't be plotted (because outside interested age range)
df_aggregate = df_aggregate[df_aggregate["age"] >= x_age_min]
df_aggregate = df_aggregate[df_aggregate["age"] <= x_age_max]

## Only use data aggregated across all races
df_aggregate = df_aggregate[df_aggregate["race_ethnicity"] == 'All']

#### CDC DATA ADD ####
df_cdc = pd.read_csv("../../data/csv/CDC_data.csv")

df_cdc = df_cdc[df_cdc["age"] >= x_age_min]
df_cdc = df_cdc[df_cdc["age"] <= x_age_max]
#################

## Group aggregate information by gender and race/ethnicity
#grouped = df_aggregate.groupby([header["gender"],header["race_ethnicity"]])
grouped = df_aggregate.groupby(["gender"])

#### CDC ADD ####
grouped_cdc = df_cdc.groupby(["gender"])
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
    if not plot_bmi_only:
        fig_dict[char]['ax_age_ht'] = fig_dict[char]['fig'].add_subplot(121)
        fig_dict[char]['ax_age_wt'] = fig_dict[char]['fig'].add_subplot(122)
    else:
        fig_dict[char]['ax_age_bmi'] = fig_dict[char]['fig'].add_subplot(111)

if not plot_bmi_only:
    attribute_pairs = [("age","ht"),("age","wt")]
else:
    attribute_pairs = [("age","bmi")]
        
### Iterate through all gender groups, plotting group percentiles
for name, group in grouped:
    name_gender = name
    #name_race_ethnicity = name[1]

    for x_attribute, y_attribute in attribute_pairs:
        color = get_color(len(percentiles))
        
        for percentile in percentiles:
            x_name = x_attribute
            y_name = y_attribute + "_" + str(percentile)
            ax_name = 'ax_'+x_attribute+'_'+y_attribute
            cat_label = str(percentile) + "%"
            cat_linewidth = 2.0
            cat_color = next(color)
            
            if percentile == 85:
                cat_linewidth = 2.0
                if plot_bmi_only:
                    cat_label = cat_label + ", Overweight"
                cat_85_color = cat_color
            elif percentile == 95:
                cat_linewidth = 2.0
                if plot_bmi_only:
                    cat_label = cat_label + ", Obese"
                cat_95_color = cat_color

            line = group.plot(x_name, y_name, ax=fig_dict[name_gender][ax_name], color=cat_color, label=cat_label, linewidth=cat_linewidth, ls="-")            

### Iterate through all gender groups, plotting CDC percentiles
for name, group in grouped:
    name_gender = name
    #name_race_ethnicity = name[1]

    for x_attribute, y_attribute in attribute_pairs:
        color = get_color(len(percentiles))
        for percentile in percentiles:
            x_name = x_attribute
            y_name = y_attribute + "_" + str(percentile)
            ax_name = 'ax_'+x_attribute+'_'+y_attribute
            cat_color = next(color)
            cat_linewidth = 2.0
            cat_label = str(percentile) + "% in subset"
            
            if display_cdc:
                line = grouped_cdc.get_group(name_gender).plot(x_name, y_name, ax=fig_dict[name_gender][ax_name], color=cat_color, linewidth=cat_linewidth, ls="--")
            
## For each figure, 
for char in fig_list:
    ## Set correct x, y axis labels
    if not plot_bmi_only:
        fig_dict[char]['ax_age_ht'].set_ylabel("Height (inches)")
        fig_dict[char]['ax_age_wt'].set_ylabel("Weight (pounds)")
        fig_dict[char]['ax_age_ht'].set_xlabel("Age (years)")
        fig_dict[char]['ax_age_wt'].set_xlabel("Age (years)")
        if not display_cdc:
            fig_dict[char]['ax_age_ht'].set_ylim([20, 80])
        fig_dict[char]['ax_age_ht'].set_xlim([x_age_min, x_age_max])
        fig_dict[char]['ax_age_wt'].set_xlim([x_age_min, x_age_max])
    else:
        fig_dict[char]['ax_age_bmi'].set_ylabel("BMI")
        fig_dict[char]['ax_age_bmi'].set_xlabel("Age (years)")
        fig_dict[char]['ax_age_bmi'].set_xlim([x_age_min, x_age_max])

    ## Add adult overweight/obesity definitions
    #fig_dict[char]['ax_age_bmi'].axhline(y=18.5, linewidth=4, color='r')
    #fig_dict[char]['ax_age_bmi'].axhline(y=25, linewidth=4, color='r')
    #fig_dict[char]['ax_age_bmi'].axhline(y=30, linewidth=4, color='r')

    #fig_dict[char]['ax_age_bmi'].plot([19.25, 20], [25, 25], linewidth=4, color=cat_85_color, ls=":", label="Adult Overweight (by definition)")
    #fig_dict[char]['ax_age_bmi'].plot([19.25, 20], [30, 30], linewidth=4, color=cat_95_color, ls=":", label="Adult Obese (by definition)")            
    #fig_dict[char]['ax_age_bmi'].plot([19.25, 19.25], [25, 30], 'D')

    ## Get legend handles and labels
    if not plot_bmi_only:
        handles, labels = fig_dict[char]['ax_age_ht'].get_legend_handles_labels()
    else:
        handles, labels = fig_dict[char]['ax_age_bmi'].get_legend_handles_labels()

    ## Invert legend order
    h1 = zip(handles, labels)
    h1 = filter(lambda (x, y): y != "age", h1)
    h1 = h1[::-1]
    handles, labels = zip(*h1)
    handles = list(handles)
    labels = list(labels)
    
    if not plot_bmi_only:
        fig_dict[char]['ax_age_ht'].legend(handles, labels, loc='top left', borderaxespad=0.)
    else:
        fig_dict[char]['ax_age_bmi'].legend(handles, labels, loc='top left', borderaxespad=0.)

    ## Set figure title
    fig_dict[char]['fig'].patch.set_facecolor('white')
    gen_dict = dict({('M', 'Male'),('F','Female')})
    fig_dict[char]['fig'].suptitle('Aggregate Patient Statistics, ' + gen_dict[char], fontsize=font_size)

    ## Set figure font sizes
    if not plot_bmi_only:
        ax_list = [fig_dict[char]['ax_age_wt'], fig_dict[char]['ax_age_ht']]
    else:
        ax_list = [fig_dict[char]['ax_age_bmi']]
        
    for ax in ax_list:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + \
                 ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(font_size)

## Show final figures
plt.show()
