# Childhood Obesity Prediction Project
This is a [Data Science for Social Good](http://www.dssg.io) project to analyze children's growth curves to determine which kids are at risk of obesity.

## The problem: increasing prevalence of obesity

Obesity is a rapidly growing epidemic in the US. More than a third of American adults are obese, and another third are overweight. If this trend holds, over 50% of American adults will be obese by 2030, making obesity the leading public health challenge of our times.

Excessive weight is bad for people’s health, wallet, and psyche. Obese adults are at increased risk for a long list of diseases, from heart disease to diabetes to cancer. Obesity also puts pressure on already tight healthcare budgets - on average, obese adults have medical bills that are 42% greater than those of normal weight adults (Finkelstein et al. 2009). 

Obesity is hard to tackle once it sinks in, so medical experts regard early prevention as the most promising method for stemming the obesity epidemic. That’s because physical exercise and good nutrition can keep children from becoming obese in the first place.

## The solution: mining growth charts for predictive tools

![NorthShore Hopsital](http://dssg.io/img/partners/northshore.jpg)

Working with NorthShore University Healthsystem in Evanston, a Chicago suburb, we wanted to determine whether we could find patterns in a child’s growth trajectory that might indicate he or she was at risk for becoming obese later on. Nearly all children have their height and weights measured and recorded on a regular basis when they visit their pediatrician. Do early warning signals exist in this data that has already been captured?

To discover these patterns, we obtained from NorthShore’s electronic medical records de-identified height, weight, and body mass index (BMI) measurements of kids as they grow up. BMI is a way to keep track of body fat, calculated from the person’s weight (in kilograms) divided by the person’s height (in meters) squared. 

In this data, we wanted to find if these children underwent a physical phenomenon called **adiposity rebound**, when children’s BMIs undergo a dip and then a rebound around age 5 to 6 (Whitaker et al., 1998; Williams and Goulding, 2009). Authors of these papers had conducted studies on a small scale showing that early adiposity rebound was associated with increased risk of adult obesity. We wanted to know if this phenomenon was present for the general population and detectable in routinely collected electronic medical records. We have posted our preliminary results in the wiki.

## The project

There are three components to the project:

### Cleaning up the growth measurements and calculating percentiles

### Visualizing the growth curves


### Analyzing the adiposity rebound for children

To install python dependencies, clone the project and run `pip install -r requirements.txt`

## The data: growth measurements of children

NorthShore provided us with the height and weight measurements of over 23,000 children de-identified records captured over the past 6 years by NorthShore hospitals’ EMR system. A sample of the data we received is shown below. We have the patients’ gender, race, and ethnicities, along with their height, weight, and BMI measurements at different ages, from the ages of 0 to 19 yrs.

|id | gender | race_ethnicity | age | bmi | ht | wt |
|------:|:-----:|:-------:|:-----:|:-----:|:---:|:---|
|1	|F 	|Caucasian	|10	|28.4	|59.3	|142|
|1	|F	|Caucasian	|11	|29.3	|61.2	|156|
|2	|M	|African American	|4.04	|17.9	|42.5	|46|
|2	|M	|African American	|5.05	|17.58	|45.6	|52|

Unfortunately, we do not have the permissions to share this medical data publicly. If you are interested in accessing the data, please email us at the email below.

## Installation 

First you will need to clone the repo. 
````
git clone https://github.com/dssg/dssg-northshore-bmi
cd dssg-northshore-bmi/
````


## Contributing to the project

To get in touch, email the team at dssg-northshore@googlegroups.com.

## References

Finkelstein, E.A., Trogdon, J.G., Cohen, J.W., and Dietz., W., 2009. Annual medical spending attributable to obesity: Payer- and service-specific estimates. Health Affairs, 28(5): w822-31.

Whitaker R.C., Pepe M.S., Wright J.A., Seidel K.D., and Dietz W.H., 1998. Early adiposity rebound and the risk of adult obesity. Pediatrics, 101(3): e5.

Williams, S.M. and Goulding, A, 2009. Patterns of growth associated with the timing of adiposity rebound. Obesity, 17(2): 335-41.

## License 

Copyright (C) 2013 [Data Science for Social Good Fellowship at the University of Chicago](http://dssg.io)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


