# -*- coding: utf-8 -*-
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

# + [markdown] magic_args="[markdown]"
# ## Topics in Pandas
# **Stats 507, Fall 2021**


# + [markdown] magic_args="[markdown]"
# ## Contents
# Add a bullet for each topic and link to the level 2 title header using 
# the exact title with spaces replaced by a dash. 
#
# # + [If-Then](#If-Then)
# # + [Time Delta](#Time-Delta)
# # + [Sorting](#Sorting)
# # + [Timestamp class](#Timestamp-class)
# # + [Table Styler](#Table-Styler)
# # + [Pandas.concat](#Pandas.concat)
# # + [Windowing Operations](#Windowing-Operations)
# # + [Interpolate](#Pandas-.interpolate()-method)
# # + [DateTime in Pandas](#DateTime-in-Pandas)
# # + [Missing Data in Pandas](#Missing-Data-in-Pandas)
# # + [Python Idioms](#Introduction-to-Python-Idioms)
# # + [Sparse Data Structures](Sparse-Data-Structures)
# # + [Pivot Table](#Pivot-Table)

# + [markdown] magic_args="[markdown]"
# ## If Then
# **Kailin Wang**
# **wkailin@umich.edu**
# -

# %%
# modules: --------------------------------------------------------------------
import numpy as np
import pandas as pd
from os.path import exists

# + [markdown] magic_args="[markdown]"
# ## Pandas `if-then`  idioms
# - The `if-then/if-then-else` idiom is a compact form of if-else that can be implemented to columns in `pd.DataFrame`
# - Expressed on one column, and assignment to another one or more columns
# - Use pandas where after you’ve set up a mask
# -


df = pd.DataFrame(
    {"AAA": [4, 5, 6, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]}
)
df

# + [markdown] magic_args="[markdown]"
# ## Pandas `if-then`  idioms
# - An `if-then` on one column
# -


df.loc[df.AAA >= 5, "BBB"] = -1
df

# + [markdown] magic_args="[markdown]"
# - An `if-then` with assignment to 2 columns:
# -


df.loc[df.AAA >= 5, ["BBB", "CCC"]] = 1022
df

# + [markdown] magic_args="[markdown]"
# ## Pandas `if-then`  idioms
# - Use pandas where after you’ve set up a mask
# -


df_mask = pd.DataFrame(
    {"AAA": [True] * 4, "BBB": [False] * 4, "CCC": [True, False] * 2}
)
df.where(df_mask,1022)

# + [markdown] magic_args="[markdown]"
# ## Pandas `if-then-else`  idioms
# - if-then-else using NumPy’s where()
# -


df = pd.DataFrame(
    {"AAA": [4, 5, 6, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]}
)
df
df["logic"] = np.where(df["AAA"] > 5, "high", "low")
df

# + [markdown] magic_args="[markdown]"
# ## Time Delta
# **Liuyu Tao**
# **liuyutao@umich.edu**

# + [markdown] magic_args="[markdown]"
# ## Overview
# - Parsing
# - to_timedelta

# + [markdown] magic_args="[markdown]"
# ## Parsing
# - There are several different methods to construct the Timeselta, below are the examples
# -


import pandas as pd
import datetime

# read as "string"
print(pd.Timedelta("2 days 3 minutes 36 seconds"))
# similar to "datetime.timedelta"
print(pd.Timedelta(days=2, minutes=3, seconds=36))
# specify the integer and the unit of the integer
print(pd.Timedelta(2.0025, unit="d"))

# + [markdown] magic_args="[markdown]"
# ## Sorting
# **Julia Weber- juliaweb@umich.edu**

# + [markdown] magic_args="[markdown]"
# ## Sorting- About
# - Pandas has built in functions that allow the user to sort values in a column or index of a dataframe.
# - Sorting is important, as a user can look for patterns in the data and easily determine which observations have the highest/lowest values for a certain variable.

# + [markdown] magic_args="[markdown]"
# ## sort_values() Function
# - The sort_values() function can be used to order rows of a dataframe by the values of a column.
# - Default sorts low to high. If we set ascending=False, sorts high to low.
# -


import pandas as pd

names = ["Julia", "James", "Andrew", "Sandy", "Joe"]
ages = [15, 18, 16, 30, 26]
test_df = pd.DataFrame({"name" : names, "age" : ages})
test_df.sort_values("age", ascending=False)

# + [markdown] magic_args="[markdown]"
# ## sort_index() Function
# - The sort_index() function can be used to sort the index of a dataframe.
# - This function is similar to the sort_values() function, but is applied to the index.
# -


sorted_df = test_df.sort_values("age", ascending=False)
sorted_df.sort_index()

# + [markdown] magic_args="[markdown]"
# ## Timestamp class
# **Yuelin He- yuelinhe@umich.edu**

# + [markdown] magic_args="[markdown]"
# Timestamp is Pandas' equivalent (and usually interchangeable) class of 
# python’s Datetime. To construct a Timestamp, there are three calling 
# conventions:
#
# 1. Converting a datetime-like string.
#
# 1. Converting a float representing a Unix epoch in units of seconds.
#
# 1. Converting an int representing a Unix-epoch in units of seconds in a 
# specified timezone.
#
# The form accepts four parameters that can be passed by position or keyword.
#
# There are also forms that mimic the API for datetime.datetime (with year, 
# month, day, etc. passed separately through parameters).
#
# See the following code for corresponding examples:
# -


import pandas as pd

## datetime-like string
print(pd.Timestamp('2021-01-01T12'))

## float, in units of seconds
print(pd.Timestamp(889088900.5, unit='s'))

##int, in units of seconds, with specified timezone
print(pd.Timestamp(5201314, unit='s', tz='US/Pacific'))

# + [markdown] magic_args="[markdown]"
# In Pandas, there are many useful attributes to do quick countings in Timestamp.
#
# - Counting the day of the...
# + year: using *day_of_year*, *dayofyear*
# - Counting the week number of the year: using *week*, *weekofyear*
# - Counting the number of days in that month: using *days_in_month*, *daysinmonth*
#
# -


# Counting the day of the week
ts = pd.Timestamp(2018, 3, 21)
print(ts.day_of_week)
print(ts.dayofweek)

# Counting the day of the year
print(ts.day_of_year)
print(ts.dayofyear)

# Counting the week number of the year
print(ts.week)
print(ts.weekofyear)

# Counting the number of days in that month
print(ts.days_in_month)
print(ts.daysinmonth)

# + [markdown] magic_args="[markdown]"
# Whether certain characteristic is true can also be determined.
#
# - Deciding if the date is the start of the...
# + month: using *is_month_start* [markdown]
# # - Similarly, deciding if the date is the end of the...
# + month: using *is_month_end* [markdown]
# # - Deciding if the year is a leap year: using *is_leap_year*
# -


# # Start?
print(pd.Timestamp(2000, 1, 1).is_year_start)
print(pd.Timestamp(2000, 2, 1).is_quarter_start)
print(pd.Timestamp(2000, 3, 1).is_month_start)

# # End?
print(pd.Timestamp(2000, 12, 31).is_year_end)
print(pd.Timestamp(2000, 12, 30).is_quarter_end)
print(pd.Timestamp(2000, 11, 30).is_month_start)

# Leap year?
print(pd.Timestamp(2000, 12, 31).is_leap_year)
print(pd.Timestamp(2001, 12, 30).is_leap_year)

# + [markdown] magic_args="[markdown]"
#
# Reference: 
# https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.html#

# + [markdown] magic_args="[markdown]"
# ## Table Styler
# ### Manipulate many parameters of a table using the table Styler object in pandas.
# **Xiying Gao**

# + [markdown] magic_args="[markdown]"
# ## Pandas.concat
# **Ziyin Chen- email: edwardzc@umich.edu**

# + [markdown] magic_args="[markdown]"
# ## General Discription
# * Concatenate pandas objects along a particular axis with optional set logic along the other axes.
#
# * Can also add a layer of hierarchical indexing on the concatenation axis, which may be useful if the labels are the same (or overlapping) on the passed axis number.

# + [markdown] magic_args="[markdown]"
# ## concat
# * used to combine tow dataframe or combining two series 
#     1. can be used to join two DataFrame or Series with or without similar column with the inclusion of `join = `
#     2. can be used to join two DataFrames either vertially or horizontally with `axis = 1`


# + [markdown] magic_args="[markdown]"
# ## Example 1 
# join two dataframe horizontaly and vertially
# -


import pandas as pd 
from IPython.display import display

dic1 = {'Name': ['Allen', 'Bill','Charle','David','Ellen'],
      'number':[1,2,3,4,5],
      'letter':['a','b','c','d','e']}
dic2 = {'A':['a','a','a','a','a'],
       'B':['b','b','b','b','b'],
       'number':[10,11,12,13,14]}
df1 = pd.DataFrame(dic1)
df2 = pd.DataFrame(dic2)
display(df1)
display(df2)

# + [markdown] magic_args="[markdown]"
# join vertially 
# -


df = pd.concat([df1,df2])
display(df)

# + [markdown] magic_args="[markdown]"
# join horizontally 
# -


df = pd.concat([df1,df2],axis =1 )
display(df)

# + [markdown] magic_args="[markdown]"
# ## Example 2 
# join with the common column
#
# -


df = pd.concat([df1,df2],join='inner')
display(df)

# + [markdown] magic_args="[markdown]"
# ## Windowing Operations
# **Mengtong Hu- mengtong@umich.edu**

# + [markdown] magic_args="[markdown]"
# - an operation that perfroms an aggregation over a sliding
#   partition of values on Series or DataFrame, similar to `groubby`.

# + [markdown] magic_args="[markdown]"
# ### Windowing Operations

# + [markdown] magic_args="[markdown]"
# - Specify the window=n argument in `.rolling()` for the window size. 
# - After specifiying the window size, apply the appropriate
#   statistical function on top of it. Examples of statistical
#   functions include: `.sum()`, `.mean()`, `.median()`, `.var()`, `.corr()`.
# - If the offest is based on a time based column such as 'window = "2D"', the correspond
#     time based index must be monotonic.
# - The example below computes the sum of 'A' for previous 2 days
# -


df = pd.DataFrame(np.arange(10),
   index = pd.date_range('1/1/2000', periods=10),
   columns = ['A'])
df['default sum'] = df['A'].rolling(window=3).sum()
df

# + [markdown] magic_args="[markdown]"
# ### Windowing Operations
# - The closed parameter in `.rolling()` is used to decide the inclusions
#     of the interval endpoints in rolling window 
#     - 'right' close right endpoint
#     - 'left' close left endpoint
#     - 'both' close both endpoints
#     - 'neither' open endpoints
# -


offset = '2D'
df["right"] = df.rolling(offset, closed="right").A.sum()  # default
df["both"] = df.rolling(offset, closed="both").A.sum()
df["left"] = df.rolling(offset, closed="left").A.sum()
df["neither"] = df.rolling(offset, closed="neither").A.sum()
df

# + [markdown] magic_args="[markdown]"
# ### Windowing Operations
# - `.apply()` function takes an extra func argument and performs self-defined rolling computations.

# + [markdown] magic_args="[markdown]"
# ## **Name**  : *Xin Luo*
# ## **EMAIL**  :  *luosanj@umich.edu*
# -
# #  Question 0
#
# ##  Pandas .interpolate() method
#
# * Method *interpolate* is very useful to fill NaN values.
# * By default, NaN values can be filled by other values with the same index for different methods.
# * Please note that NaN values in DataFrame/Series with MultiIndex can be filled by 'linear' method as
# <code>method = 'linear' </code>. 

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

m =  pd.Series([0, 1, np.nan, np.nan, 3, 5, 8])
m.interpolate(method = 'pad', limit = 2)

n = pd.Series([10, 2, np.nan, 4, np.nan, 3, 2, 6]) 
n.interpolate(method = 'polynomial', order = 2)


# ##### parameter **'axis'** :  default *None*
# * 1. axis = 0 : Axis to interpolate along is index.
# * 2. axis = 1 : Axis to interpolate along is column.
#     

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

pd.date_range(start='2020-01-01', end='2020-01-05', freq='D')

pd.date_range(start='2020-01-01', end='2021-01-01', freq='2M')


# ## Convert from other types
#
# - Other list-like objects like strings can also be easily converted to a pandas `DatetimeIndex` using `to_datetime` function. This function can infer the format of the string and convert automatically.

pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-05"])


# - A `format` keyword argument can be passed to ensure specific parsing.

pd.to_datetime(["2020/01/01", "2020/01/03", "2020/01/05"], format="%Y/%m/%d")


# ## Indexing with DatetimeIndex
#
# - One of the main advantage of using the `DatetimeIndex` is to make index a time series a lot easier. For example, we can use common date string to directly index a part of the time series.

idx = pd.date_range('2000-01-01', '2021-12-31', freq="M")
ts = pd.Series(np.random.randn(len(idx)), index=idx)

ts['2018-09':'2019-04']

ts['2021-6':]


# ## Date/time components in the DatetimeIndex
#
# - The properties of a date, e.g. `year`, `month`, `day_of_week`, `is_month_end` can be easily obtained from the `DatetimeIndex`.

idx.isocalendar()


# ## Operations on Datetime
#
# - We can shift a DatetimeIndex by adding or substracting a `DateOffset`

idx[:5] + pd.offsets.Day(2)

idx[:5] + pd.offsets.Minute(1)

get_ipython().system('/usr/bin/env python')
# coding: utf-8


# ### youngwoo Kwon
#
# kedwaty@umich.edu

# # Question 0 - Topics in Pandas

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

df = pd.DataFrame(
    np.random.randn(4, 3),
    index=["a", "c", "e", "f"],
    columns=["one", "two", "three"],
)
df["five"] = df["one"] < 0
df2 = df.reindex(["a", "b", "c", "d", "e"])
df2

df2.isna()

df2.notna()


# ## More about the Missing Values
#
# * In Python, nan's don't compare equal, but None's do.
#
# * NaN is a float, but pandas provides a nullable integer array

None == None

np.nan == np.nan

print(df2.iloc[1,1])
print(type(df2.iloc[1,1]))

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

get_ipython().system('/usr/bin/env python')
# coding: utf-8

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

import pandas as pd
df = pd.DataFrame({"A": [1, 3, 5, 7, 9], "B": [10, 20, 30, 40, 50], "C": [100, 200, 300, 400, 500]})
df


# * To assign to one or more column:

df.loc[df.A > 5, 'B'] = '> 5'
df


# * or

df.loc[df.A > 5, ['B','C']] = '> 5'
df


# * You can add another line with different logic, to do the ”-else“

df.loc[df.A <= 5, ['B','C']] = '< 5'
df


# 1.2 You can also apply "if-then-else" using Numpy's where( ) function

import numpy as np
import pandas as pd

df = pd.DataFrame({"A": [1, 3, 5, 7, 9], "B": [10, 20, 30, 40, 50], "C": [100, 200, 300, 400, 500]})
df['new'] = np.where(df['A'] > 5, '> 5', '< 5')
df


# ## 2. Splitting a frame with a boolean criterion

# You can split a data frame with a boolean criterion

import numpy as np
import pandas as pd

df = pd.DataFrame({"A": [1, 3, 5, 7, 9], "B": [10, 20, 30, 40, 50], "C": [100, 200, 300, 400, 500]})
df

df[df['A'] > 5]

df[df['A'] <= 5]


# ## 3. Building criteria 
# You can build your own selection criteria using "**and**" or "**or**".  
#
# 3.1 "... and"

import numpy as np
import pandas as pd

df = pd.DataFrame({"A": [1, 3, 5, 7, 9], "B": [10, 20, 30, 40, 50], "C": [100, 200, 300, 400, 500]})
df


# * ...and

df.loc[(df["B"] < 25) & (df["C"] >= 20), "A"]


# * ...or

df.loc[(df["B"] < 25) | (df["C"] >= 40), "A"]


# * you can also assign new value to a existing column using this method

df.loc[(df["B"] > 40) | (df["C"] >= 300), "A"] = 'new'
df


# ## Takeaways  
# There are a few python idioms that can help speeding up your data managemet.  
# * "if-then-else" allows you easily change the current column or add additional new columns based on the value of a specific column
# * "Splitting" allows you quickly select specific rows based on the value of a specific column
# * "Building criteria" allows you select specific data from one column or assign new values to one column based on the criteria you set up on other columns

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

s = pd.Series(pd.arrays.SparseArray([1] * 2 + [np.nan] * 8))
s


# `Sparse[float64, nan]` indicates that all values apart from `nan` are stored,
#  and they are of type float.

# ### Memory Efficiency
# - The `memory_usage` function can be used to inspect the number of bytes
# being consumed by the Series/DataFrame
# - Comparing memory usage between a SparseArray and a regular python list
# represented as a Series depicts the memory efficiency of SparseArrays

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

pd.pivot_table(recs2015, index = 'REGIONC',aggfunc={'NWEIGHT':np.mean,'CDD65':np.sum, 'HDD65':np.sum})


# ### Pivot table: aggfunc functionality
#  - Aggregate on specific features with values parameter
#  - Meanwhile, can use mulitple aggfunc via a list

pd.pivot_table(recs2015, index = 'REGIONC', values = 'NWEIGHT', aggfunc = [np.mean, len])


# ### Pivot table: find how data is correlated
#  - Find relationship between feature with columns parameter
#  - UATYP10 - A categorical data type representing census 2010 urban type
#  -         U: Urban Area; R: Rural; C: Urban Cluster

pd.pivot_table(recs2015,index='REGIONC',columns='UATYP10',values='NWEIGHT',aggfunc=np.sum)


# ## Takeaways

#  - Pivot Table is beneficial when we want to have a deep investigation on the dataset. <br>
#  - Especially when we want to find out where we can explore more from the dataset. <br>
#  - Meanwhile, the aggfunc it involves effectively minimizes our work. <br>
