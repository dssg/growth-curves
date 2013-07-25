##### SETUP ######

import sys
sys.path.append('../config')

import config
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import interpolate

header = config.header
attributes = config.attributes

## Add additional headers into header dictoinary
header["count"] = "Number of Datapoints"
header["ar_find"] = "Method of Finding AR Point"
header["age_ar"] = "Age at AR"
header["age_end"] = "Age at End"

## header_list used later in creating new rows
header_list = [header["id"], header["gender"], header["race_ethnicity"], \
               header["count"], header["ar_find"], header["age_ar"], header["age_end"]]

for attribute in attributes.keys():
    header["perc_"+attribute+"_ar"] = "Percentile "+attributes[attribute]+" at AR"
    header["perc_"+attribute+"_end"] = "Percentile "+attributes[attribute]+" at End"
    header_list = header_list + [header["perc_"+attribute+"_ar"], header["perc_"+attribute+"_end"]]

##################

pkl_file = open('../../data/pkl/BMI_ar_calculate_5pt.pkl', 'rb')
df_orig = pickle.load(pkl_file)

good_cases = [4,5]
df = df_orig

agg_bool = df.shape[0]*[False]
for case in good_cases:
    agg_bool = np.logical_or(agg_bool, df[header["ar_find"]] == case)
df = df[agg_bool]
fig = plt.figure()
fig.add_subplot(111)
ax = fig.add_subplot(111)
ax.plot(df[header["age_ar"]],df[header["perc_bmi_end"]], 'ro')
ax.set_xlabel("Age at Adiposity Rebound (year)")
ax.set_ylabel("BMI Percentile % at End of Individual's Growth Curve")

#ax2 = fig.add_subplot(122)
#ax2.plot(df[header["perc_bmi_ar"]],df[header["perc_bmi_end"]], 'ro')
#ax2.set_xlabel("BMI Percentile % at AR")
#ax2.set_ylabel("BMI Percentile % at end")

num_x = df.shape[0]
x = df[header["age_ar"]]
x2 = df[header["perc_bmi_ar"]]
X = np.column_stack((x, np.ones(num_x)))
y = df[header["perc_bmi_end"]]
model = sm.RLM(y,X)
results = model.fit()
b = results.params
xnew = np.linspace(x.min(),x.max(),100)
ynew = b[0]*xnew + b[1]
#ax.plot(xnew,ynew, 'b-')

model = sm.OLS(y,X)
results = model.fit()
b = results.params
ynew = b[0]*xnew + b[1]
ax.plot(xnew,ynew, 'g-')

plt.show()
print results.summary()

grouped_ar_find = df_orig.groupby(header["ar_find"])
counts = grouped_ar_find[header["id"]].count()
percentages = grouped_ar_find[header["id"]].count()*100.0/df_orig.shape[0]

bmi_deriv_desc = ["BMI deriv > 0", \
"BMI deriv < 0", \
"BMI deriv starts > 0, ends < 0", \
"BMI deriv starts < 0, ends > 0", \
"BMI curve starts > 0, dips < 0, ends > 0", \
"BMI curve starts < 0, dips > 0, ends < 0"]

bmi_desc = ["BMI constantly increases", \
"BMI constantly decreases", \
"BMI passes through max", \
"BMI passes through min", \
"BMI passes through local min", \
"BMI passes through local max"]

print pd.DataFrame({"Counts":counts, "Percentages":percentages, \
                    "BMI Deriv":bmi_deriv_desc, "BMI":bmi_desc})
