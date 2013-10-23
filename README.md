# Childhood obesity prediction

[![NorthShore Hopsital](http://dssg.io/img/partners/northshore.jpg)](http://www.northshore.org/)

Statistical models of children's growth curves that predict which kids are at risk of obesity.

This is project is a part of the 2013 [Data Science for Social Good](http://www.dssg.io) fellowship, in partnership with [NorthShore Hospital](http://www.northshore.org/).

## The problem: obesity epidemic

Obesity is a rapidly growing epidemic in the US. More than a third of American adults are obese, and another third are overweight. If this trend holds, 42% of American adults will be obese by 2030 (Finkleson et al., 2012), making obesity the leading public health challenge of our time.

Excessive weight is bad for people’s health, wallet, and psyche. Obese adults are at increased risk for a long list of diseases, from heart disease to diabetes to cancer. Obesity also puts pressure on already tight healthcare budgets - on average, obese adults have medical bills that are 42% greater than those of normal weight adults (Finkelstein et al. 2009). 

Obesity is hard to tackle once it sinks in, so medical experts regard early prevention as the most promising method for stemming the obesity epidemic. That’s because physical exercise and good nutrition can keep children from becoming obese in the first place.

But how can we predict which kids are at risk of becoming obese as they grow, so that we can intervene before it's too late?

**[To learn more about the obesity epidemic and early prevention, read the wiki.](https://github.com/dssg/growth-curves/wiki/problem)**

## The solution: predictive modeling using electronic medical records

Working with NorthShore University Healthsystem in Evanston, a Chicago suburb, we wanted to determine whether we could find patterns in a child’s growth trajectory that might indicate he or she was at risk for becoming obese later on. Nearly all children have their height and weights measured and recorded on a regular basis when they visit their pediatrician. Do early warning signals exist in this ubiquitous and routinely collected data?

To discover these patterns, we obtained anonimized electronic medical records from NorthShore on the height, weight, and body mass index (BMI) measurements of young patients over the years. (BMI is a way to keep track of body fat, calculated from the person’s weight (in kilograms) divided by the person’s height (in meters) squared.)

We wanted to detect if any of these children experienced a physical phenomenon called **adiposity rebound**, when a child's BMI dips and rebounds between age 5 to 6 (Whitaker et al., 1998; Williams and Goulding, 2009). Small-scale studies have suggested that early adiposity rebound is associated with increased risk of adult obesity. So we wanted to know if this phenomenon was present in the NortShore's patient population, and detectable in routinely collected electronic medical records. 

**[For more on our methodology, read the wiki.](https://github.com/dssg/growth-curves/wiki/methodology)**

## Project layout

There are three components to the project in `code`:

### Cleaning up the growth measurements and calculating percentiles

In `code/init_processing/`, we first clean up NortShore's EMR growth measurements by keeping individuals with enough datapoints and removing extreme outliers. We then aggregate the measurements into percentiles for each gender. We also isolate subsets of the population that we are interested in (e.g. kids who were obese at age 5).

### Visualizing the growth curves

In `code/visualize`, we plot the percentiles in the form of growth charts. We can also compare our growth charts to growth charts from the Center for Disease Control (CDC).

### Analyzing the adiposity rebound for children

In `code/adiposity_rebound`, we attempt to detect adiposity rebound in all NortShore patients whose growth curves extend past the age of 5 (due to available EMR data). We then run linear regressions and find that age at adiposity rebound is a statistically significant predictor of a child's final BMI percentile, and thus of their obesity risk.

## The data: growth measurements of children

NorthShore provided us with the height and weight measurements of over 23,000 de-identified children. NorthShore's EMR systems captured these records over the past 6 years. The data looks like this: 

|id | gender | race_ethnicity | age | bmi | ht | wt |
|------:|:-----:|:-------:|:-----:|:-----:|:---:|:---|
|1	|F 	|Caucasian	|10	|28.4	|59.3	|142|
|1	|F	|Caucasian	|11	|29.3	|61.2	|156|
|2	|M	|African American	|4.04	|17.9	|42.5	|46|
|2	|M	|African American	|5.05	|17.58	|45.6	|52|

We have the patients’ gender, race, and ethnicities, along with their height, weight, and BMI measurements at different ages, from the ages of 0 to 19 yrs.

Due to the sensitive nature of this medical data, we aren't able to share it publicly. If you're interested in working with us, please get in touch.

**[To learn more about EMR data, read the wiki.](https://github.com/dssg/growth-curves/wiki/data)**

However, we are able to provide the height, weight, and BMI data the U.S. Center for Disease Control and Prevention (CDC) uses for their growth charts. This reference data can be found at `data/csv/CDC_data.csv`

## Additional folders

`figures/for_wiki/` contains figures for the [wiki](https://github.com/dssg/growth-curves/wiki/) on our project. 

## Installation guide

First you will need to clone the repo. 
````
git clone https://github.com/dssg/growth-curves
cd growth-curves/
````

Then you will need to install the python dependencies by running 
`pip install -r requirements.txt`

## Team

<img src="https://github.com/dssg/dssg-northshore-bmi/blob/master/figures/for_wiki/northshore_team.png?raw=true" align="center">

## Contributing to the project

To get in touch, email the team at dssg-northshore@googlegroups.com.

## References

Finkelstein, E.A., Trogdon, J.G., Cohen, J.W., and Dietz., W., 2009. Annual medical spending attributable to obesity: Payer- and service-specific estimates. Health Affairs, 28(5): w822-31.

Finkelstein, E.A., Khavjou, O.A., Thompson, H., Trogdon, J.G., Pan, L., Sherry, B., and Dietz, W., 2012. Obesity and severe obesity forecasts through 2030. American Journal of Preventive Medicine, 42(6): 563-570.

Whitaker R.C., Pepe M.S., Wright J.A., Seidel K.D., and Dietz W.H., 1998. Early adiposity rebound and the risk of adult obesity. Pediatrics, 101(3): e5.

Williams, S.M. and Goulding, A, 2009. Patterns of growth associated with the timing of adiposity rebound. Obesity, 17(2): 335-41.

## License 

Copyright (C) 2013 [Data Science for Social Good Fellowship at the University of Chicago](http://dssg.io)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


