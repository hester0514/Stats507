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
# ##### 24107190
# ##### STATS507 PS3

import pandas as pd
import numpy as np
import os
import requests
import math
from IPython.core.display import display, HTML
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
from urllib import request

# ### Question 0 - RECS and Replicate Weights

# ##### Data Files

recs2015 = pd.read_csv("https://www.eia.gov/consumption/residential/data/2015/csv/recs2015_public_v4.csv")
recs2009 = pd.read_csv("https://www.eia.gov/consumption/residential/data/2009/csv/recs2009_public.csv")
recs_weight = pd.read_csv("https://www.eia.gov/consumption/residential/data/2009/csv/recs2009_public_repweights.csv")

# ##### Variables

#  - "DOEID": **-numeric-** Unique identifier for each respondent
#  - "NWEIGHT": **-numeric-**Final sample weight
#  - "REGIONC": **-categorical-**Census Region
#  - "CDD30YR": **-numeric-**Cooling degree days, 30-year average 1981-2010, base temperature 65F
#  - "CDD65": **-numeric-**Cooling degree days in 2015, base temperature 65F
#  - "CDD80": **-numeric-**Cooling degree days in 2015, base temperature 80F
#  - "HDD30YR":**-numeric-**Heating degree days, 30-year average 1981-2010, base temperature 65F
#  - "HDD65":**-numeric-**Heating degree days in 2015, base temperature 65F
#  - "HDD50": **-numeric-**Heating degree days in 2015, base temperature 50F<br>
#  
#  
#  - **we may only use CDD65 an HDD65 for calculations in the Question 2 which contain the true meaning toward the variance calculation**
#  - **The CDDxx and HDDxx is still used in Question 1, but it is just for constructing dataframe and will not influence the question**

# ##### Weights and Replicate Weights

# **2015GuideBook**:[Compute Estimate and Standard Errors](https://www.eia.gov/consumption/residential/data/2015/pdf/microdata_v3.pdf).<br>
#
# **2009GuideBook**:[Compute Estimate and Standard Errors](https://www.eia.gov/consumption/residential/methodology/2009/pdf/using-microdata-022613.pdf).

# The **standard error** is calculated by applying Fay's method of balanced repeated replication (BRR) to estimate the statistic of interest repeatly and find the differences from the full-sample estimate.<br>
#
# **$\theta$** refers to the population parameter of interest and let **$\hat{\theta}$** be the estimate from the full sample of $\theta$. <br>
#
# Fay coefficient, $\epsilon$, a range in-between $0\le\epsilon\lt1$.<br>
# In this scenario, $\epsilon$ refers to 0.5 for both 2009 and 2015 cases.<br>
#
#
# Let $\hat{\theta_r}$ be the **estimate** from the r-th replicate subsample by using replicate weights.<br>
#
# **r** refers to the **sample order of the replicate weights**; In 2015, there are 96 BRRWT replicate weights from 1st to 96th, while in 2009, there are 244 BRR replicate weights from 1st to 244th. They are used later in calculating standard error of replicates weights from full sample weights.
#
# The **variance** of $\hat{\theta}$ is <br>
#
# $\hat(\theta) = \frac{1}{R(1-\epsilon)^2}\sum_{r=1}^R(\hat{\theta_r} - \hat{\theta})^2$

# #### Question 1 - Data Preparation

#Check if file exists in the local
path1 = '/Users/Sylvia/Desktop/recs2015_public_v4.csv'
path2 = '/Users/Sylvia/Desktop/recs2009_public.csv'
path3 = '/Users/Sylvia/Desktop/recs2009_public_repweights.csv'
is_exists1 = os.path.exists(path1)
is_exists2 = os.path.exists(path2)
is_exists3 = os.path.exists(path3)
if is_exists1 == True:
    recs2015 = pd.read_csv(path1)
else:
    recs2015 = pd.read_csv("https://www.eia.gov/consumption/residential/data/2015/csv/recs2015_public_v4.csv")
if is_exists2 == True:
    recs2009 = pd.read_csv(path2)
else:
    recs2009 = pd.read_csv("https://www.eia.gov/consumption/residential/data/2009/csv/recs2009_public.csv")
if is_exists3 == True:
    recs_weight = pd.read_csv(path3)
else:
    recs_weight = pd.read_csv("https://www.eia.gov/consumption/residential/data/2009/csv/recs2009_public_repweights.csv")

# ##### part a)

#Extract columns from dataset
recs2015_v = recs2015[['DOEID','NWEIGHT','REGIONC','CDD30YR', 'CDD65', 'CDD80', 'HDD30YR','HDD65','HDD50']]

#Rename columns
recs2015_v.rename(columns = {"DOEID": "id","NWEIGHT":"samp_wei",
                             "REGIONC":"census_re"}, inplace = True)
recs2015_v.rename(columns = {"CDD30YR": "cd_30yrs",
                             "CDD65":"cd65","CDD80":"cd80"}, inplace = True)
recs2015_v.rename(columns = {"HDD30YR": "hd_30yrs",
                             "HDD65":"hd65","HDD50":"hd50"}, inplace = True)
recs2015_v

#Substitute categorical data
categ_name = {"census_re":{1:"Northeast", 2: "Midwest", 3: "South", 4: "West"}}
recs2015_v = recs2015_v.replace(categ_name)
recs2015_v

#Repeat for 2009
recs2009_v = recs2009[['DOEID','NWEIGHT','REGIONC','CDD30YR', 'CDD65', 'CDD80', 'HDD30YR','HDD65','HDD50']]
recs2009_v.rename(columns = {"DOEID": "id","NWEIGHT":"samp_wei","REGIONC":"census_re"}, inplace = True)
recs2009_v.rename(columns = {"CDD30YR": "cd_30yrs","CDD65":"cd65","CDD80":"cd80"}, inplace = True)
recs2009_v.rename(columns = {"HDD30YR": "hd_30yrs","HDD65":"hd65","HDD50":"hd50"}, inplace = True)
recs2009_v = recs2009_v.replace(categ_name)
recs2009_v

#DataType of 2015
recs2015_v['id'] = recs2015_v['id'].astype(int)
recs2015_v['samp_wei'] = recs2015_v['samp_wei'].round(3).astype(float)
recs2015_v['census_re'] = recs2015_v['census_re'].astype('category')
recs2015_v.iloc[:,3:] = recs2015_v.iloc[:,3:].astype(int)

#DataType of 2009
recs2009_v['id'] = recs2009_v['id'].astype(int)
recs2009_v['samp_wei'] = recs2009_v['samp_wei'].round(3).astype(float)
recs2009_v['census_re'] = recs2009_v['census_re'].astype('category')
recs2009_v.iloc[:,3:] = recs2009_v.iloc[:,3:].astype(int)

# ##### part b)

#  - 2015 Replicate Weight

#Extract BRRWT from 2015 datasets
recs_weight_15 = recs2015[['DOEID','NWEIGHT']]
checkcol = recs2015.loc[:, recs2015.columns.str.startswith('BRRWT')]
recs_weight_15 = pd.concat([recs_weight_15, checkcol], axis = 1)

#Rename replicate weights data in 2015
recs_weight_15 = recs_weight_15.rename(columns = {"DOEID":"id", "NWEIGHT":"samp_wei"})

#brrwt_mean - mean of the 96 brrwt values, it is irrelated to the problem 
#(deleting it will influence the dataset that I used later, 
#therefore, just leave it here)
recs_weight_15["brrwt_mean"] = recs_weight_15.iloc[:,1:].mean(axis = 1)

#Collect column names of all replicate weights
col = recs_weight_15.columns.tolist()
col.remove("id")
col.remove("samp_wei")
col.remove("brrwt_mean")

#  - Long Format for ids and duplicate weights in 2015

#Long Format of 2015
longformat_15 = recs_weight_15["id"]
longformat_15 = pd.concat([longformat_15, recs_weight_15[col]], axis = 1)
longformat_15.stack().to_frame(name = "BRRWTvalue")

#  - 2009 Replicate Weight

recs_weight_09 = pd.merge(recs_weight, recs2009, on = "DOEID")
recs_weight_09 = recs_weight_09.iloc[:, 0:246]
recs_weight_09 = recs_weight_09.drop(columns = "NWEIGHT_x")

recs_weight_09 = recs_weight_09.rename(columns = {"DOEID":"id"})

#  - Long format for ids and duplicate weights in 2009

#Long Format of 2009
recs_weight_09.stack().to_frame()

# #### Question 2 - Construct and report the estimates

# ##### part a)

# ## 2015

#Merge datafile with weight file, and rename
recs_w_2015 = pd.merge(recs2015_v, recs_weight_15, on = 'id')
recs_w_2015 = recs_w_2015.drop(columns = "samp_wei_y")
recs_w_2015 = recs_w_2015.rename(columns = {"samp_wei_x":"samp_wei"})
recs_w_2015.head()

#  - **CD65:** Point Estimate of final weight of 2015 by region.

#Compute Point Estimate of final weight of cooling days in 2015
recs_w_2015["sw_cd65"] = recs_w_2015["samp_wei"]*recs_w_2015["cd65"]
sw_cd_65sum_15 = recs_w_2015.groupby("census_re")["sw_cd65"].sum().reset_index(name = "sum")
sum_nweight15 = recs_w_2015.groupby("census_re")["samp_wei"].sum().reset_index(name = "nweightsum")
sw_cd_65sum_15["point_est"] = (sw_cd_65sum_15["sum"]/sum_nweight15["nweightsum"]).round(2)
sw_cd_65sum_15

#Compute sum by regions
brrwt = pd.concat([recs_w_2015["census_re"],recs_w_2015[col]],axis = 1)
brrwt = brrwt.groupby("census_re").sum().reset_index()
categ_ = {"census_re":{"Northeast":"NortheastS", 
                       "Midwest":"MidwestS", "South":"SouthS", "West":"WestS"}}
brrwt = brrwt.replace(categ_)
brrwt

# +
#Multiply with cooling days
recs_w_2015val = recs_w_2015.iloc[:,9:-2].multiply(recs_w_2015["cd65"], axis = "index")
recs_w_2015val = pd.concat([recs2015_v, recs_w_2015val], axis = 1)

#Group by census region
group_cd65_15 = recs_w_2015val.groupby("census_re")[col].agg("sum")
group_cd65_15 = group_cd65_15.reset_index()
group_cd65_15 = group_cd65_15.iloc[:,1:]

#Divide by the sum of brrwt
brrwt = brrwt.iloc[:,1:]
final_cd_15 = group_cd65_15/brrwt
sd_cd_15 = pd.concat([sw_cd_65sum_15,final_cd_15], axis = 1)
sub_col_cd15 = sd_cd_15.iloc[:,3:]

#Subtract theta_r from theta_hat and square the result
sub_col_cd15 = sub_col_cd15.sub(sd_cd_15["point_est"], axis = 0)
sub_col_cd15 = sub_col_cd15.pow(2)
sub_col_cd15["sum_byregionc"] = sub_col_cd15.sum(axis = 1)
sum_byregionccd15 = sub_col_cd15["sum_byregionc"].reset_index()

#Apply the variance computation formula
result_cd_15 = sum_byregionccd15["sum_byregionc"]/(96*(1-0.5)**2)
result_cd_15 = result_cd_15.reset_index(name = "theta_hat")
_name = {"index":{0:"Midwest", 1: "Northeast", 2: "South", 3: "West"}}

#Calculate the standard error
result_cd_15 = result_cd_15.replace(_name)
result_cd_15["sd"] = result_cd_15["theta_hat"]**(1/2)
result_cd_15 = result_cd_15.rename(columns = {"index":"census_re"})
result_cd_15
# -

#  - **Confidence Interval** for CDD65 in 2015

#Calculate confidence interval of Cooling Days in 2015
zscore = 1.96
ci_cd15 = pd.merge(result_cd_15, sw_cd_65sum_15, on = "census_re")
ci_cd15["lower_bound"] = (ci_cd15["point_est"] - ci_cd15["sd"]*zscore).round(2)
ci_cd15["upper_bound"] = (ci_cd15["point_est"] + ci_cd15["sd"]*zscore).round(2)
ci_cd15["CI"] = "("+(ci_cd15["lower_bound"]).astype(str) + "," +(ci_cd15["upper_bound"]).astype(str)+")"
ci_cd15

#  - **HD65:** Point Estimate of final weight of 2015 by region.

#Compute Point Estimate of Heating Days in 2015 by region
recs_w_2015["sw_hd65"] = recs_w_2015["samp_wei"]*recs_w_2015["hd65"]
sw_hd_65sum_15 = recs_w_2015.groupby("census_re")["sw_hd65"].sum().reset_index(name = "sum")
sum_nweight15 = recs_w_2015.groupby("census_re")["samp_wei"].sum().reset_index(name = "nweightsum")
sw_hd_65sum_15["point_est"] = (sw_hd_65sum_15["sum"]/sum_nweight15["nweightsum"]).round(2)
sw_hd_65sum_15

#  - **standard error** of replicate weights of heating days in 2015

# +
#Multiply with heating days
recs_w_2015val = recs_w_2015.iloc[:,9:-2].multiply(recs_w_2015["hd65"], axis = "index")
recs_w_2015val = pd.concat([recs2015_v, recs_w_2015val], axis = 1)

#Group by census region
group_hd65_15 = recs_w_2015val.groupby("census_re")[col].agg("sum")
group_hd65_15 = group_hd65_15.reset_index()
group_hd65_15 = group_hd65_15.iloc[:,1:]

#Divide by the sum of brrwt
brrwt = brrwt.iloc[:,1:]
final_hd_15 = group_hd65_15/brrwt
sd_hd_15 = pd.concat([sw_hd_65sum_15,final_hd_15], axis = 1)
sub_col_hd15 = sd_hd_15.iloc[:,3:]

#Subtract theta_r from theta_hat and square the result
sub_col_hd15 = sub_col_hd15.sub(sd_hd_15["point_est"], axis = 0)
sub_col_hd15 = sub_col_hd15.pow(2)
sub_col_hd15["sum_byregionc"] = sub_col_hd15.sum(axis = 1)
sum_byregionchd15 = sub_col_hd15["sum_byregionc"].reset_index()

#Apply the variance computation formula
result_hd_15 = sum_byregionchd15["sum_byregionc"]/(96*(1-0.5)**2)
result_hd_15 = result_hd_15.reset_index(name = "theta_hat")
_name = {"index":{0:"Midwest", 1: "Northeast", 2: "South", 3: "West"}}

#Calculate the standard error
result_hd_15 = result_hd_15.replace(_name)
result_hd_15["sd"] = result_hd_15["theta_hat"]**(1/2)
result_hd_15 = result_hd_15.rename(columns = {"index":"census_re"})
result_hd_15
# -

#  - **Confidence Interval** for heating days in 2015

#Compute the final confidence interval
zscore = 1.96
ci_hd15 = pd.merge(result_hd_15, sw_hd_65sum_15, on = "census_re")
ci_hd15["lower_bound"] = (ci_hd15["point_est"] - ci_hd15["sd"]*zscore).round(2)
ci_hd15["upper_bound"] = (ci_hd15["point_est"] + ci_hd15["sd"]*zscore).round(2)
ci_hd15["CI"] = "("+(ci_hd15["lower_bound"]).astype(str) + "," +(ci_hd15["upper_bound"]).astype(str)+")"
ci_hd15

# ## 2009

#  - **CD65**: Point Estimate of final weight of 2009 by region

#Merge datafile with weight file
recs_w_2009 = pd.merge(recs2009_v, recs_weight_09, on = "id")
recs_w_2009.head()

#Compute Point Estimate of NWEIGHT in 2009 by regions
recs_w_2009["sw_cd65"] = recs_w_2009["samp_wei"]*recs_w_2009["cd65"]
sw_cd_65sum_09 = recs_w_2009.groupby("census_re")["sw_cd65"].sum().reset_index(name = "sum")
sum_nweight09 = recs_w_2009.groupby("census_re")["samp_wei"].sum().reset_index(name = "nweightsum")
sw_cd_65sum_09["point_est"] = (sw_cd_65sum_09["sum"]/sum_nweight09["nweightsum"]).round(2)
sw_cd_65sum_09

#  - **HD65**: Point Estimate of final weight of 2009 by region

#Compute Point Estimate of Heating Days in 2009 by regions
recs_w_2009["sw_hd65"] = recs_w_2009["samp_wei"]*recs_w_2009["hd65"]
sw_hd_65sum_09 = recs_w_2009.groupby("census_re")["sw_hd65"].sum().reset_index(name = "sum")
sum_nweight09 = recs_w_2009.groupby("census_re")["samp_wei"].sum().reset_index(name = "nweightsum")
sw_hd_65sum_09["point_est"] = (sw_hd_65sum_09["sum"]/sum_nweight09["nweightsum"]).round(2)
sw_hd_65sum_09

#Collect weights' column names in a list
col09 = recs_weight_09.columns.tolist()
col09.remove("id")

#Calculate sum of Nweights by regions
brr = pd.concat([recs_w_2009["census_re"],recs_w_2009[col09]],axis = 1)
brr = brr.groupby("census_re").sum().reset_index()
cate_ = {"census_re":{"Northeast":"NortheastS", 
                      "Midwest":"MidwestS", "South":"SouthS", "West":"WestS"}}
brr = brr.replace(cate_)
brr

#  - **CD65**: Confidence Interval computation of cooling days of 2009 by region

# +
#Multiply with cooling days
recs_w_2009val = recs_w_2009.iloc[:,9:-2].multiply(recs_w_2009["cd65"], axis = "index")
recs_w_2009val = pd.concat([recs2009_v, recs_w_2009val], axis = 1)

#Group by census region
group_cd65_09 = recs_w_2009val.groupby("census_re")[col09].agg("sum")
group_cd65_09 = group_cd65_09.reset_index()
group_cd65_09 = group_cd65_09.iloc[:,1:]
brr = brr.iloc[:,1:]

#Divide by the sum of brr
final_cd_09 = group_cd65_09/brr
sd_cd_09 = pd.concat([sw_cd_65sum_09,final_cd_09], axis = 1)
sub_col_cd09 = sd_cd_09.iloc[:,3:]

#Subtract theta_r from theta_hat and square the result
sub_col_cd09 = sub_col_cd09.sub(sd_cd_09["point_est"], axis = 0)
sub_col_cd09 = sub_col_cd09.pow(2)
sub_col_cd09["sum_byregionc"] = sub_col_cd09.sum(axis = 1)
sum_byregionccd09 = sub_col_cd09["sum_byregionc"].reset_index()

#Apply the variance computation formula
result_cd_09 = sum_byregionccd09["sum_byregionc"]/(244*(1-0.5)**2)
result_cd_09 = result_cd_09.reset_index(name = "theta_hat")
_name = {"index":{0:"Midwest", 1: "Northeast", 2: "South", 3: "West"}}
result_cd_09 = result_cd_09.replace(_name)

#Calculate the standard error
result_cd_09["sd"] = result_cd_09["theta_hat"]**(1/2)
result_cd_09 = result_cd_09.rename(columns = {"index":"census_re"})
result_cd_09
# -

#Compute the final confidence interval
zscore = 1.96
ci_cd09 = pd.merge(result_cd_09, sw_cd_65sum_09, on = "census_re")
ci_cd09["lower_bound"] = (ci_cd09["point_est"] - ci_cd09["sd"]*zscore).round(2)
ci_cd09["upper_bound"] = (ci_cd09["point_est"] + ci_cd09["sd"]*zscore).round(2)
ci_cd09["CI"] = "("+(ci_cd09["lower_bound"]).astype(str) + "," +(ci_cd09["upper_bound"]).astype(str)+")"
ci_cd09

#  - **HD65**: Confidence Interval computation of heating days of 2009 by region

# +
#Multiply with cooling days
recs_w_2009val = recs_w_2009.iloc[:,9:-2].multiply(recs_w_2009["hd65"], axis = "index")
recs_w_2009val = pd.concat([recs2009_v, recs_w_2009val], axis = 1)

#Group by census region
group_hd65_09 = recs_w_2009val.groupby("census_re")[col09].agg("sum")
group_hd65_09 = group_hd65_09.reset_index()
group_hd65_09 = group_hd65_09.iloc[:,1:]
brr = brr.iloc[:,1:]

#Divide by the sum of brr
final_hd_09 = group_hd65_09/brr
sd_hd_09 = pd.concat([sw_hd_65sum_09,final_hd_09], axis = 1)
sub_col_hd09 = sd_hd_09.iloc[:,3:]

#Subtract theta_r from theta_hat and square the result
sub_col_hd09 = sub_col_hd09.sub(sd_hd_09["point_est"], axis = 0)
sub_col_hd09 = sub_col_hd09.pow(2)
sub_col_hd09["sum_byregionc"] = sub_col_hd09.sum(axis = 1)
sum_byregionchd09 = sub_col_hd09["sum_byregionc"].reset_index()

#Apply the variance computation formula
result_hd_09 = sum_byregionchd09["sum_byregionc"]/(244*(1-0.5)**2)
result_hd_09 = result_hd_09.reset_index(name = "theta_hat")
_name = {"index":{0:"Midwest", 1: "Northeast", 2: "South", 3: "West"}}
result_hd_09 = result_hd_09.replace(_name)

#Calculate the standard error
result_hd_09["sd"] = result_hd_09["theta_hat"]**(1/2)
result_hd_09 = result_hd_09.rename(columns = {"index":"census_re"})
result_hd_09
# -

#Compute the confidence interval of heating days in 2009
zscore = 1.96
ci_hd09 = pd.merge(result_hd_09, sw_hd_65sum_09, on = "census_re")
ci_hd09["lower_bound"] = (ci_hd09["point_est"] - ci_hd09["sd"]*zscore).round(2)
ci_hd09["upper_bound"] = (ci_hd09["point_est"] + ci_hd09["sd"]*zscore).round(2)
ci_hd09["CI"] = "("+(ci_hd09["lower_bound"]).astype(str) + "," +(ci_hd09["upper_bound"]).astype(str)+")"
ci_hd09

#Summary data in a table
ci_cd15_ = ci_cd15[["point_est","CI"]]
ci_hd15_ = ci_hd15[["point_est","CI"]]
ci_cd09_ = ci_cd09[["point_est","CI"]]
ci_hd09_ = ci_hd09[["point_est","CI"]]
summ = {'2015CDD' : ci_cd15_, '2015HDD' : ci_hd15_,'2009CDD':ci_cd09_,'2009HDD':ci_hd09_}
summary = pd.concat(summ.values(), axis=1, keys=summ.keys())
summary

#  - HTML format for 2a

cap = """
<b> Table 2a.</b> <em> Summary table of heating and cooling days in 2009 and 2015
"""
t1 = summary.to_html(index=False)
t1 = t1.rsplit('\n')
t1.insert(1, cap)
tab1 = ''
for i, line in enumerate(t1):
    tab1 += line
    if i < (len(t1) - 1):
        tab1 += '\n'

display(HTML(tab1))

# ##### part b)

# +
#95% 
zscore = 1.96

#Construct Tables and do calculations
table = ci_hd09['census_re'].reset_index()
table["point_est_hdd"] = ci_hd15["point_est"] - ci_hd09["point_est"]
table["point_est_cdd"] = ci_cd15["point_est"] - ci_cd09["point_est"]
table["var_hdd"] = ci_hd15["theta_hat"] + ci_hd09["theta_hat"]
table["var_cdd"] = (ci_cd15["theta_hat"] + ci_cd09["theta_hat"])
table["sd_hdd"] = ((table["var_hdd"])**(1/2))
table["sd_cdd"] = ((table["var_cdd"])**(1/2))

#Calculate Upper and Lower Bounds
table["up_cdd"] = (table["point_est_cdd"]+zscore*table["sd_cdd"]).round(2)
table["lo_cdd"] = (table["point_est_cdd"]-zscore*table["sd_cdd"]).round(2)
table["up_hdd"] = (table["point_est_hdd"]+zscore*table["sd_hdd"]).round(2)
table["lo_hdd"] = (table["point_est_hdd"]-zscore*table["sd_hdd"]).round(2)
table["CI_hdd"] = "("+(table["lo_hdd"]).astype(str)+" , "+(table["up_hdd"]).astype(str)+")"
table["CI_cdd"] = "("+(table["lo_cdd"]).astype(str)+" , "+(table["up_cdd"]).astype(str)+")"
table
# -

#Move to a new table
updated_sum = table[["point_est_hdd","CI_hdd","point_est_cdd","CI_cdd"]]
updated_sum = updated_sum.rename(columns = {"point_est_hdd":"AvgHeatingDays",
                                           "CI_hdd":"CI_HeatingDays",
                                           "point_est_cdd":"AvgCoolingDays",
                                           "CI_cdd":"CI_CoolingDays"})
updated_sum

#  - HTML format for 2b

cap = """
<b> Table 2b.</b> <em> Difference between 2009 and 2015 in cooling and heating scenarios.
"""
t2 = summary.to_html(index=False)
t2 = t2.rsplit('\n')
t2.insert(1, cap)
tab2 = ''
for i, line in enumerate(t2):
    tab2 += line
    if i < (len(t2) - 1):
        tab2 += '\n'

display(HTML(tab2))

# #### Question 3 Visualizations

# ##### matplot refers to Q2 part a

# +
#Q2 part a: 2009 VS 2015 Cooling Days
fig1, ax1 = plt.subplots(nrows=1, ncols=1)
_ = plt.scatter(
    data=ci_cd09,
    x="point_est",
    y="census_re",
    marker='s',
    color='green'
    )
_ = plt.errorbar(
    x=ci_cd09["point_est"],
    y=ci_cd09["census_re"],
    fmt='None',
    xerr=(ci_cd09["sd"] * 1.96, ci_cd09["sd"] * 1.96),
    ecolor='green',
    capsize=4
    )
_ = plt.scatter(
    data=ci_cd15,
    x="point_est",
    y="census_re",
    marker='s',
    color='red'
    )
_ = plt.errorbar(
    x=ci_cd15["point_est"],
    y=ci_cd15["census_re"],
    fmt='None',
    xerr=(ci_cd15["sd"] * 1.96, ci_cd15["sd"] * 1.96),
    ecolor='red',
    capsize=4
    )

_ = ax1.legend(["2009","2015"],loc='upper left') 
_ = ax1.set_xlabel('AverageCoolingDays: Point Estimation and 95% CI (4 Regions)')
# -

# This error bar plot consists point estimation and confidence interval of Cooling Days in both 2009 and 2015, we can see 2015 obtain a larger confidence interval for four different regions, simultaneously, 2009 obtain a lower average cooling days than 2015, it may indicates there's a negative sign happens in 2015 during cooling days.

#Q2 Part a: 2009 vs 2015 Heating Days
fig3, ax3 = plt.subplots(nrows=1, ncols=1)
_ = plt.scatter(
    data=ci_hd09,
    x="point_est",
    y="census_re",
    marker='s',
    color='red'
    )
_ = plt.errorbar(
    x=ci_hd09["point_est"],
    y=ci_hd09["census_re"],
    fmt='None',
    xerr=(ci_hd09["sd"] * 1.96, ci_hd09["sd"] * 1.96),
    ecolor='red',
    capsize=4
    )
_ = plt.scatter(
    data=ci_hd15,
    x="point_est",
    y="census_re",
    marker='s',
    color='green'
    )
_ = plt.errorbar(
    x=ci_hd15["point_est"],
    y=ci_hd15["census_re"],
    fmt='None',
    xerr=(ci_hd15["sd"] * 1.96, ci_hd15["sd"] * 1.96),
    ecolor='green',
    capsize=4
)
_ = ax3.legend(["2009","2015"],loc='upper right') 
_ = ax3.set_xlabel('AverageHeatingDays: point estimation and 95% CI (4 Regions)')

# This error bar plot consists point estimation and confidence interval of heating days in both 2009 and 2015. Compare with the previous cooling days graph, 2015 is currently having a lower point estimation value in heating days compared to 2009. Therefore, the weather may getting cooler as time passed by.

# ##### Matplot refers to Q2 part b

#Q2 Part b: Cooling Days
fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(10,4))
_ = plt.scatter(
    data=table,
    x="point_est_cdd",
    y="census_re",
    marker='s',
    color='blue'
    )
_ = plt.errorbar(
    x=table["point_est_cdd"],
    y=table["census_re"],
    fmt='None',
    xerr=(table["sd_cdd"] * 1.96, table["sd_cdd"] * 1.96),
    ecolor='blue',
    capsize=4
    )
_ = plt.scatter(
    data=table,
    x="point_est_hdd",
    y="census_re",
    marker='s',
    color='red'
    )
_ = plt.errorbar(
    x=table["point_est_hdd"],
    y=table["census_re"],
    fmt='None',
    xerr=(table["sd_hdd"] * 1.96, table["sd_hdd"] * 1.96),
    ecolor='red',
    capsize=4
)
_ = ax4.legend(["CoolingDays","HeatingDay"],loc='right') 
_ = ax4.set_xlabel('AverageCoolingDays: Point Estimation and 95% CI (4 Regions)')

# This is the plot of point estimation and confidence interval of difference in 2009 and 2015. We can see the difference is obvious which indicates the weather has a great change.
