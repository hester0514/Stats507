#!/usr/bin/env python
# coding: utf-8

# ## **Name**  : *Xin Luo*
# ## **EMAIL**  :  *luosanj@umich.edu*

# #  Question 0
# 
# ##  Pandas .interpolate() method
# 
# * Method *interpolate* is very useful to fill NaN values.
# * By default, NaN values can be filled by other values with the same index for different methods.
# * Please note that NaN values in DataFrame/Series with MultiIndex can be filled by 'linear' method as
# <code>method = 'linear' </code>. 

# In[ ]:


import pandas as pd
import numpy as np
a = pd.DataFrame({'a' : [1, 2, np.nan, 5], 'b' : [4, np.nan, 6, 8]})
a.interpolate(method = 'linear')


# ### Parameters in .interpolate()
# ##### *parameter **'method'** : *str*, default *'linear'
# 
# 
# * Most commonly used methods:
#     * 1. **'linear'** : linear regression mind to fit the missing ones.
#     * 2. **'pad', 'limit'** :  Fill in NaNs using existing values. Note:Interpolation through padding means copying the value just before a missing entry.While using padding interpolation, you need to specify a limit. The limit is the maximum number of nans the method can fill consecutively.
#     * 3. **'polynomial', 'order'** : Polynomial regression mind with a set order to fit the missing ones. Note : NaN of the first column remains, because there is no entry before it to use for interpolation.

# In[ ]:


m =  pd.Series([0, 1, np.nan, np.nan, 3, 5, 8])
m.interpolate(method = 'pad', limit = 2)


# In[ ]:


n = pd.Series([10, 2, np.nan, 4, np.nan, 3, 2, 6]) 
n.interpolate(method = 'polynomial', order = 2)


# ##### parameter **'axis'** :  default *None*
# * 1. axis = 0 : Axis to interpolate along is index.
# * 2. axis = 1 : Axis to interpolate along is column.
#     

# In[ ]:


k = pd.DataFrame({'a' : [1, 2, np.nan, 5], 'b' : [4, np.nan, 6, 8]})
k.interpolate(method = 'linear', axis = 0)
k.interpolate(method = 'linear', axis = 1)


# ###  Returns
# * Series or DataFrame or None
# * Returns the same object type as the caller, interpolated at some or all NaN values or None if `inplace=True`.

# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     notebook_metadata_filter: markdown
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Topics in Pandas
# **Stats 507, Fall 2021** 
#   

# ## Contents
# 
# + [DateTime in Pandas](#DateTime-in-Pandas) 
# + [Create DatetimeIndex](#Create-DatetimeIndex) 
# + [Convert from other types](#Convert-from-other-types) 
# + [Indexing with DatetimeIndex](#Indexing-with-DatetimeIndex) 
# + [Date/time components in the DatetimeIndex](#Date/time-components-in-the-DatetimeIndex) 
# + [Operations on Datetime](#Operations-on-Datetime) 

# ## DateTime in Pandas
# 
# *Qi, Bingnan*
# bingnanq@umich.edu
# 
# - Pandas contains a collection of functions and features to deal with time series data. A most commonly used class is `DatetimeIndex`.
# 

# ## Create DatetimeIndex
# 
# - A `DatetimeIndex` array can be created using `pd.date_range()` function. The `start` and `end` parameter can control the start and end of the range and `freq` can be `D` (day), `M` (month), `H` (hour) and other common frequencies.

# In[ ]:


pd.date_range(start='2020-01-01', end='2020-01-05', freq='D')


# In[ ]:


pd.date_range(start='2020-01-01', end='2021-01-01', freq='2M')


# ## Convert from other types
# 
# - Other list-like objects like strings can also be easily converted to a pandas `DatetimeIndex` using `to_datetime` function. This function can infer the format of the string and convert automatically.

# In[ ]:


pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-05"])


# - A `format` keyword argument can be passed to ensure specific parsing.

# In[ ]:


pd.to_datetime(["2020/01/01", "2020/01/03", "2020/01/05"], format="%Y/%m/%d")


# ## Indexing with DatetimeIndex
# 
# - One of the main advantage of using the `DatetimeIndex` is to make index a time series a lot easier. For example, we can use common date string to directly index a part of the time series.

# In[ ]:


idx = pd.date_range('2000-01-01', '2021-12-31', freq="M")
ts = pd.Series(np.random.randn(len(idx)), index=idx)

ts['2018-09':'2019-04']


# In[ ]:


ts['2021-6':]


# ## Date/time components in the DatetimeIndex
# 
# - The properties of a date, e.g. `year`, `month`, `day_of_week`, `is_month_end` can be easily obtained from the `DatetimeIndex`.

# In[ ]:


idx.isocalendar()


# ## Operations on Datetime
# 
# - We can shift a DatetimeIndex by adding or substracting a `DateOffset`

# In[ ]:


idx[:5] + pd.offsets.Day(2)


# In[ ]:


idx[:5] + pd.offsets.Minute(1)


# In[ ]:


get_ipython().system('/usr/bin/env python')
# coding: utf-8


# ### youngwoo Kwon
# 
# kedwaty@umich.edu

# # Question 0 - Topics in Pandas

# In[1]:

# In[ ]:


import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import time


# ## Missing Data in Pandas 
# 
# Pandas is very __flexible__ to the missing values
# 
# * NaN is the default missing value
# 
# * However, we should deal with the different types such as integer, boolean, or general object.
# 
# * We should also consider that "missing" or "not available" or "NA".

# ## Detecting the Missing Values
# 
# * Pandas provides `isna()` and `notna()` function to detect the missing values

# In[2]:

# In[ ]:


df = pd.DataFrame(
    np.random.randn(4, 3),
    index=["a", "c", "e", "f"],
    columns=["one", "two", "three"],
)
df["five"] = df["one"] < 0
df2 = df.reindex(["a", "b", "c", "d", "e"])
df2


# In[3]:

# In[ ]:


df2.isna()


# In[4]:

# In[ ]:


df2.notna()


# ## More about the Missing Values
# 
# * In Python, nan's don't compare equal, but None's do.
# 
# * NaN is a float, but pandas provides a nullable integer array

# In[5]:

# In[ ]:


None == None


# In[6]:

# In[ ]:


np.nan == np.nan


# In[7]:

# In[ ]:


print(df2.iloc[1,1])
print(type(df2.iloc[1,1]))


# In[8]:

# In[ ]:


pd.Series([1, 2, np.nan, 4], dtype=pd.Int64Dtype())


# ## Topics in Pandas
# **Stats 507, Fall 2021** 
#   

# ## Contents
# Add a bullet for each topic and link to the level 2 title header using 
# the exact title with spaces replaced by a dash. 
# 
# + [Introduction to Python Idioms](https://github.com/boyazh/Stats507/blob/main/pandas_notes/pd_topic_boyazh.py) 
# + [Topic 2 Title](#Topic-2-Title)

# ## Topic Title
# Python Idioms
# **Boya Zhang**

# In[ ]:


get_ipython().system('/usr/bin/env python')
# coding: utf-8


# In[ ]:





# # Question 0

# ## Introduction to Python Idioms  
#   
# Boya Zhang (boyazh@umich.edu)  
# 
# 10.16.21
# 

# ## Overview  
#   
# 1. if-then/if-then-else
# 2. splitting
# 3. building criteria

# ## 1. if-then/ if-then-else 
# 
# 1.1 You can use if-then to select specific elements on one column, and add assignments to another one or more columns: 
#         

# In[21]:

# In[ ]:


import pandas as pd
df = pd.DataFrame({"A": [1, 3, 5, 7, 9], "B": [10, 20, 30, 40, 50], "C": [100, 200, 300, 400, 500]})
df


# * To assign to one or more column:

# In[22]:

# In[ ]:


df.loc[df.A > 5, 'B'] = '> 5'
df


# * or

# In[23]:

# In[ ]:


df.loc[df.A > 5, ['B','C']] = '> 5'
df


# * You can add another line with different logic, to do the ”-else“

# In[25]:

# In[ ]:


df.loc[df.A <= 5, ['B','C']] = '< 5'
df


# 1.2 You can also apply "if-then-else" using Numpy's where( ) function

# In[28]:

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


df = pd.DataFrame({"A": [1, 3, 5, 7, 9], "B": [10, 20, 30, 40, 50], "C": [100, 200, 300, 400, 500]})
df['new'] = np.where(df['A'] > 5, '> 5', '< 5')
df


# ## 2. Splitting a frame with a boolean criterion

# You can split a data frame with a boolean criterion

# In[38]:

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


df = pd.DataFrame({"A": [1, 3, 5, 7, 9], "B": [10, 20, 30, 40, 50], "C": [100, 200, 300, 400, 500]})
df


# In[39]:

# In[ ]:


df[df['A'] > 5]


# In[40]:

# In[ ]:


df[df['A'] <= 5]


# ## 3. Building criteria 
# You can build your own selection criteria using "**and**" or "**or**".  
# 
# 3.1 "... and"

# In[49]:

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


df = pd.DataFrame({"A": [1, 3, 5, 7, 9], "B": [10, 20, 30, 40, 50], "C": [100, 200, 300, 400, 500]})
df


# * ...and

# In[50]:

# In[ ]:


df.loc[(df["B"] < 25) & (df["C"] >= 20), "A"]


# * ...or

# In[51]:

# In[ ]:


df.loc[(df["B"] < 25) | (df["C"] >= 40), "A"]


# * you can also assign new value to a existing column using this method

# In[52]:

# In[ ]:


df.loc[(df["B"] > 40) | (df["C"] >= 300), "A"] = 'new'
df


# ## Takeaways  
# There are a few python idioms that can help speeding up your data managemet.  
# * "if-then-else" allows you easily change the current column or add additional new columns based on the value of a specific column
# * "Splitting" allows you quickly select specific rows based on the value of a specific column
# * "Building criteria" allows you select specific data from one column or assign new values to one column based on the criteria you set up on other columns

# In[ ]:

# ## Topic Title
# Python Idioms
# **Xi Zheng**

# ## 1. pandas.Series.ne
#  * Return Not equal to of series and other, element-wise (binary operator ne).
#  * `Series.ne(other, level=None, fill_value=None, axis=0)`
#  - Parameters:  
#      - otherSeries or scalar value
#      - fill_valueNone or float value, default None (NaN)
#      - levelint or name
#      - Returns: series
# 
# ## 2. Code Example
# ```python
# a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
# b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
# a.ne(b, fill_value=0)
# ```
# ## 3. Ouput and Explanation
# ```
# a :
# 
# a    1.0
# b    1.0
# c    1.0
# d    NaN
# ```
# 
# ```
# b: 
# 
# a    1.0
# b    NaN
# d    1.0
# e    NaN
# ```
# 
# ```
# a    False
# b     True
# c     True
# d     True
# e     True
# ```
# ## Takeaways
# The function tells the equivalence of corresponding elements in a and b, where the 'True' means 'not equal', and 'False' means 'equal'.

# # Topics in Pandas
# 
# ## Sparse Data Structures
# 
# #### Author: Chittaranjan (chitt@umich.edu)

# In[ ]:


# imports
import pandas as pd
import numpy as np


# ### Sparse Data Structures
# - Pandas provides a way of efficiently storing "sparse" data structures
# - A sparse data structure is one in which a majority of the values are
# omitted (to be interpreted as 0, or NaN, or any other value)
# - It can be thought of as a "compressed" representation, as all values are
# not explicitly stored in the data structure

# ### Creating a Sparse Data Frame
# - Sparse data frames can be created using `pd.arrays.SparseArray`
# - Has a dtype of `Sparse` which has two values associated with it,
#     - Type of non-omitted values (Eg: float, int etc)
#     - Value of the elements in the array that aren't actually stored
# (Eg: 0, nan)
# 

# In[ ]:


s = pd.Series(pd.arrays.SparseArray([1] * 2 + [np.nan] * 8))
s


# `Sparse[float64, nan]` indicates that all values apart from `nan` are stored,
#  and they are of type float.

# ### Memory Efficiency
# - The `memory_usage` function can be used to inspect the number of bytes
# being consumed by the Series/DataFrame
# - Comparing memory usage between a SparseArray and a regular python list
# represented as a Series depicts the memory efficiency of SparseArrays

# In[ ]:


N = 1000  # number of elements to be represented

proportions = list(range(100, N+1, 100))
sparse_mems = []
non_sparse_mems = []
for proportion in proportions:
    sample_list = [14] * proportion + [np.nan] * (N - proportion)
    sparse_arr = pd.Series(
        pd.arrays.SparseArray(sample_list)
    )
    sparse_mem = sparse_arr.memory_usage()
    sparse_mems.append(sparse_mem)

    non_sparse_arr = pd.Series(sample_list)
    non_sparse_mem = non_sparse_arr.memory_usage()
    non_sparse_mems.append(non_sparse_mem)

x = list(map(lambda p: p / N, proportions))
_ = plt.plot(x, non_sparse_mems)
_ = plt.plot(x, sparse_mems)
_ = plt.ylabel("Memory Usage (bytes)")
_ = plt.xlabel("Proportion of values")
_ = plt.legend(["Non-Sparse", "Sparse"])
_ = plt.title("Comparison of Memory Usage (Size=1000)")


# ### Memory Efficiency (Continued)
# - The Sparse Arrays consume much less memory when the density is low
# (sparse-ness is high)
# - As the density increases to where 50-60% of the values are not nan
# (i.e ommittable), memory efficiency is worse

# ## Pivot Table

# **Stats 507, Fall 2021**

# **Xuechun Wang** <br>
# **24107190** <br>
# **xuechunw@umich.edu** <br>

# ## How Pivot table works

# - Pivot table is a table of grouped value that aggregates individual items of a more extensive table.
#  - The aggregations can be count, sum, mean, stats tool etc.
#  - Levels can be stored as multiIndex objects and columns of the result DataFrame.
#  - It arrange or rearrange data to provide a more direct insight into datasets
#  - **pd.pivot_table(data, values = None, index = None, aggfunc = 'mean'...)** can take more parameters
#  - Requires data and index parameter, data is the dataFrame passed into the function, index allow us to group the data 

# In[2]:


#import packages
import pandas as pd
import numpy as np

#import used dataset as example
recs2015 = pd.read_csv("https://www.eia.gov/consumption/residential/data/2015/csv/recs2015_public_v4.csv")

#Pivot table can take single or multiple indexes via a list to see how data is grouped by
pd.pivot_table(recs2015, index = ['REGIONC','DOEID'])


# ### Pivot table: mean and sum calculation
#  - Apply different aggregation function for different feature
#  - We can calculate mean of NWEIGHT and sum of CDD65 after groupbying regions

# In[3]:


pd.pivot_table(recs2015, index = 'REGIONC',aggfunc={'NWEIGHT':np.mean,'CDD65':np.sum, 'HDD65':np.sum})


# ### Pivot table: aggfunc functionality
#  - Aggregate on specific features with values parameter
#  - Meanwhile, can use mulitple aggfunc via a list

# In[4]:


pd.pivot_table(recs2015, index = 'REGIONC', values = 'NWEIGHT', aggfunc = [np.mean, len])


# ### Pivot table: find how data is correlated
#  - Find relationship between feature with columns parameter
#  - UATYP10 - A categorical data type representing census 2010 urban type
#  -         U: Urban Area; R: Rural; C: Urban Cluster

# In[5]:


pd.pivot_table(recs2015,index='REGIONC',columns='UATYP10',values='NWEIGHT',aggfunc=np.sum)


# ## Takeaways

#  - Pivot Table is beneficial when we want to have a deep investigation on the dataset. <br>
#  - Especially when we want to find out where we can explore more from the dataset. <br>
#  - Meanwhile, the aggfunc it involves effectively minimizes our work. <br>
