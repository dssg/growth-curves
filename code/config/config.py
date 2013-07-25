import numpy as np
import pandas as pd

attributes = {"ht":"Height", "wt":"Weight", "bmi":"BMI"}

## Set up age intervals
## Float instability created errors, so corrected the following trunction 
intervals = np.trunc(np.concatenate((np.array([0.01]),np.arange(.05,.2,.05),np.arange(.4,2,.2), np.arange(2,20,.5)))*100)/100

percentiles = np.array([.03, .05, .1, .25, .50, .75, .85, .90, .95, .97])

def convert_percentiles(percentiles):
    percentiles_int = np.round(percentiles*100).astype(int).tolist()
    num_percentiles = len(percentiles_int)
    return percentiles_int, num_percentiles

## Set up column headers
header = dict()
header["id"] = "Patient ID"
header["gender"] = "Gender"
header["race_ethnicity"] = "Race/Ethnicity"
header["age"] = "Patient Age At Encounter"
header["count"] = "Num of Individiuals"
header["bmi"] = "Body Mass Index"
header["wt_ounces"] = "Weight (Ounces)"
header["wt"] = "Weight (Pounds)"
header["ht"] = "Height (Inches)"

header_list_base = [header["gender"], header["age"], header["race_ethnicity"],header["count"]]

def add_attributes_to_header_dict(attributes, header, header_list, percentiles): 
    percentiles_int, num_percentiles = convert_percentiles(percentiles)

    for attribute in attributes.keys():
        header[attribute + "_mean"] = attributes[attribute] + " Mean"
        header[attribute + "_std"] = attributes[attribute] + " std"
        header_list = header_list + [header[attribute + "_mean"], header[attribute + "_std"]]
        
        for percentile in percentiles_int:
            header[attribute + "_" + str(percentile)] = attributes[attribute] + ", " + str(percentile) + "%"
            header_list = header_list + [header[attribute + "_" + str(percentile)]]
           
    return (header, header_list)

header, header_list = add_attributes_to_header_dict(attributes, header, header_list_base, percentiles)

## Color generator with maximum spacing between colors
## from: http://stackoverflow.com/questions/10254207/color-and-line-writing-using-matplotlib
import colorsys

def get_color(color):
    for hue in range(color):
        hue = 1. * hue / color
        col = [int(x) for x in colorsys.hsv_to_rgb(hue, 1.0, 230)]
        yield "#{0:02x}{1:02x}{2:02x}".format(*col)


