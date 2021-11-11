#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# umich id: xuechunw
# umich email: xuechunw@umich.edu

# ## Pivot Table

#  - Pivot table is a table of grouped value that aggregates individual items of a more extensive table.
#  - The aggregations can be count, sum, mean, stats tool etc.
#  - Levels can be stored as multiIndex objects and columns of the result DataFrame.
#  - It arrange or rearrange data to provide a more direct insight into datasets
#  - **pd.pivot_table(data, values = None, index = None, aggfunc = 'mean'...)** can take more parameters
#  - Requires data and index parameter, data is the dataFrame passed into the function, index allow us to group the data 

# In[2]:


#import used dataset as example
recs2015 = pd.read_csv("https://www.eia.gov/consumption/residential/data/2015/csv/recs2015_public_v4.csv")

#Pivot table can take single or multiple indexes via a list to see how data is grouped by
pd.pivot_table(recs2015, index = ['REGIONC','DOEID'])


# In[3]:


#Apply different aggregation function for different feature
#We can calculate mean of NWEIGHT and sum of CDD65 after groupbying regions
pd.pivot_table(recs2015, index = 'REGIONC',aggfunc={'NWEIGHT':np.mean,'CDD65':np.sum, 'HDD65':np.sum})


# In[4]:


#Aggregate on specific features with values parameter
#Meanwhile, can use mulitple aggfunc via a list
pd.pivot_table(recs2015, index = 'REGIONC', values = 'NWEIGHT', aggfunc = [np.mean, len])


# In[5]:


#Find relationship between feature with columns parameter
#UATYP10 - A categorical data type representing census 2010 urban type
#        - U: Urban Area; R: Rural; C: Urban Cluster
pd.pivot_table(recs2015,index='REGIONC',columns='UATYP10',values='NWEIGHT',aggfunc=np.sum)

