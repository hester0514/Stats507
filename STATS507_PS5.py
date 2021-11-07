# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# #### Xuechun Wang
# #### STATS 507 PS5
# #### 24107190

import pandas as pd
import numpy as np
from os.path import exists
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# ## Question 0 - R-Squared Warmup

#read in datafile
file = 'tooth_growth.feather'
if exists(file):
    tg_data = pd.read_feather(file)
else: 
    tooth_growth = sm.datasets.get_rdataset('ToothGrowth')
    tg_data = tooth_growth.data
    tg_data.to_feather(file)

# +
#transform dataset and get dummy categorical variable
tg_data['log_len'] = tg_data[['len']].transform(np.log)
tg_data['dose_cat'] = pd.Categorical(tg_data['dose'])
Y = tg_data['log_len']
X = pd.get_dummies(tg_data['supp'])['OJ']

#buile linear regression model for independent variables
mod1 = sm.OLS.from_formula('Y ~ X + dose_cat', data=tg_data)
res1 = mod1.fit()
res1.summary2()
# -

#build linear regression model for interaction term
mod3 = smf.ols('Y ~ X*dose_cat', data=tg_data)
res3 = mod3.fit()
res3.summary2()

# We applied OLS to fit a linear regression model for response variable (with transformation) and independent variables of supplement types and categroical variable, doses (model 1), and another model for interaction terms (model 2). From the summary table, we see R squared and Adjusted R are 0.751 and 0.738 for model 1 respectively, and for model 2 is 0.794 and 0.775 respectively. It indicates about 75% variation in Y is explained by the independent variables and the model 2 has a higher R-squared compared to model 1, which may suggest model 2 is more reliable in covering the variations.

# ## Question 1 - NHANES Dentition

# ##### part a

#read in data file
ohx = pd.read_feather("ohx.feather")
demo = pd.read_pickle("demo_updated.pickle")

#levels for categorical variables
demo_cat = {
    'gender': {1: 'Male', 2: 'Female'},
    'race': {1: 'Mexican American',
             2: 'Other Hispanic',
             3: 'Non-Hispanic White',
             4: 'Non-Hispanic Black',
             6: 'Non-Hispanic Asian',
             7: 'Other/Multiracial'
             },
    'education': {1: 'Less than 9th grade',
                  2: '9-11th grade (Includes 12th grade with no diploma)',
                  3: 'High school graduate/GED or equivalent',
                  4: 'Some college or AA degree',
                  5: 'College graduate or above',
                  7: 'Refused',
                  9: "Don't know"
                  },
    'martial_status': {1.0: 'Married',
                       2.0: 'Widowed',
                       3.0: 'Divorced',
                       4.0: 'Separated',
                       5.0: 'Never married',
                       6.0: 'Living with partner',
                       77.0: 'Refused',
                       99.0: "Don't know"
                       },
    'exam_status': {1.0: 'Interviewed only',
                    2.0: 'Both interviewed and MEC examined'
                    }
    }

#replace dataset
demo = demo.replace(demo_cat)

#merge and clean dataset
result = pd.merge(ohx, demo, on = "id", how = "left")
result['tc_07'] = result['tc_07'].dropna()
X = result['age']

#get dummy for all the categorical variables
tc = ['tc_' + str(i).zfill(2) for i in range(1,33)]
response = ['y_' + str(i).zfill(2) for i in range(1,33)]
y = {column: None for column in response}
for i in range(32):
    y.update({response[i]: 1 * (result[tc[i]] == "Permanent tooth present")})
df = pd.DataFrame(y).reset_index()
df = pd.concat([df, result], axis = 1)
table = df.groupby('age')['y_07'].mean().reset_index(name = "probability")

#plot the graph for the selected tooth
plt.title("B-spline basis (degree=3)")
plt.plot(table['age'], table['probability'])

#use knots to generate bspline and use aic to diagonoise it
Y = df['y_07']
knots1 = (12, 13, 40, 60)
knots2 = (11, 15, 23, 35)
knots3 = (9, 10, 12, 15)
teeth_mod = smf.logit(
    'Y~bs(age, knots=knots3, degree=3)+gender+'+'race', data=df)
fit_result = teeth_mod.fit()
fit_result.aic

# knots3 has the best performance, therefore, in this case, we apply knot3 to fit the data.

# ##### part b

#select all the tooth variables
result_age = df.loc[:,'age']
tclist = df.loc[:, 'y_01':'y_32']
tclist = pd.concat([result_age, tclist], axis = 1)

#apply logistic regression on all the teeth variables
temp_list = []
for i in range(1, 33):
    knots3 = (9, 10, 12, 15)
    y_ = response[i-1]
    reg = smf.logit(y_+'~bs(age, knots=knots3, degree=3)', data=tclist)
    fit = reg.fit(method='bfgs')
    temp_list.append((reg, fit))
    num = str(i).zfill(2)
    yhat = 'yhat_'+ num
    tclist[yhat] = reg.predict(fit.params)

#group by the dataset based on age factor
tc_prob = tclist.groupby('age').mean().reset_index()
tc_prob

# ##### part c

# +
#plot the predicted variable for all the teeth variables
age = [i for i in range(80)]
_tc_ = tc_prob.loc[:, "yhat_01":]

fig, axes = plt.subplots(8,4)
fig.set_size_inches(20, 25)
count = 0
for i in range(8):
    for j in range(4):
        axes[i, j].plot(age, _tc_.iloc[:,count])
        count += 1
# -

# ## Question 2 - Hosmer-Lemeshow Calibration Plot

# ##### part a

#divide the dataset into 10 percentiles
tc_selected = tclist[['y_07', 'yhat_07']]
tc_selected['Decile'] = pd.qcut(tc_selected['yhat_07'], 10, labels=False)

# ##### part b

#obtain the mean for predicted data and observed data
tc_mean = tc_selected.groupby('Decile').mean()
tc_mean

# ##### part c

#plot the calibration with slope = 1
fig2, ax2 = plt.subplots(nrows=1, ncols=1)
_ = plt.scatter(
    data=tc_mean,
    x='y_07',
    y='yhat_07',
    marker='s',
    color='black'
    )
point1 = [0, 0]
point2 = [1, 1]
x_values = [point1[0], point2[0]]
y_values = [point1[1], point2[1]]
plt.plot(x_values, y_values, color = 'red')

# ##### part d

# The plot looks great with the data. The data points follow the slope of 1 and show a well-calibrated graph. Therefore, we can conclude that the calibration plot is suffciently conclude all the datapoints and show a great linear trend.
