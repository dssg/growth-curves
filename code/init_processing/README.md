`init_processing.py` cleans up the growth measurements by keeping individuals with enough datapoints and removing extreme outliers. 

To determine outlier datapoints, the code performs regression for each individual. For most individuals, the code performs a cubic regression of the log of the height (or weight) against the age. However, for patients that only have data collected after age 15, the code performs a linear regression of the log of height (or weight) against the age. This is because kids do not grow as much (or at all) here, so great changes in the height (and to a lesser extent, weight) are not expected. Outliers are residuals with p-values less than 1% in a 2-sided T-test.

`calc_quartiles.py` calculates the percentile lines for each gender. 

`calc_subset_quartiles.py` first isolates a subset of the population that we are interested in (e.g. kids who were obese at age 5). It then calculates the percentiles lines for each gender in this subset.

`aggregate.py` contains the function `calculate_aggregations`, which calculates the percentile lines for each given group. Both `calc_quartiles.py` and `calc_subset_quartiles.py` call this function.
