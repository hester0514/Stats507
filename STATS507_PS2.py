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

# ### Xuechun Wang
# #### STATS 507 Problem Set 2
# #### 24107190

import pandas as pd
import numpy as np
from collections import defaultdict
from operator import itemgetter
from timeit import Timer
import pickle
import random
from itertools import chain
from IPython.core.display import display, HTML

# ### Q0

# #### a.) 
#  - After revising part of the code(by indenting the 2nd nested for loop and replacing 3 to 2 in the position index), the code snippet can successfully run. The entire program is about comparing the element in the list and conclude a new list. The second element (0,1,2) fails to compare with the rest two and automatically join into the new list op. The first and third element hold the same initial number and distinct last number, therefore, they went to the second for loop, and after sorting, (1,9,8) remains in the op.

# #### b.)
#
#  - The indention and position index is important when we go over an entire list, incorrect indentation will mislead the entire meaning of the loop, and similarly, inappropriate position index will lead to a different number, and cause the entire program goes wrong or stop the running.<br>
#  
#
#  - For the second nested for loop, start comparison at the second tuple. Therefore, it can save the running time for repeatly compare the first tuple with the rest again. Use m+1 to refer to the next tuple from the last for loop. <br>
#  
#  
#  - A clear variable name will help recalling what it is used for.<br>
#  

sample_list = [(1, 3, 5), (0, 1, 2), (1, 9, 8)]
op = []
for m in range(len(sample_list)):
    li = [sample_list[m]]
    for n in range(len(sample_list)):
        if (sample_list[m][0] == sample_list[n][0] and
                sample_list[m][2] != sample_list[n][2]):
            li.append(sample_list[n])
            
    op.append(sorted(li, key=lambda dd: dd[2], reverse=True)[0])  
res = list(set(op))
print("res=",res)


# ### Q1

def generateTuples(n, k, tmin, tmax):   
    '''
    

    Parameters
    ----------
    n : integer
        size of the array - describe how many tuples will be in the array.
    k : integer
        size of the tuple - descrieb how many integers will be in the tuple.
    tmin : integer
        minimum number of initiating a random tuple.
    tmax : integer
        maximum number of initaiting a random tuple.

    Returns
    -------
    li_output : a list of tuple
        the output will be a randomized list of array with maximum number, minimum number and range has been set.

    '''
        
    output = np.random.randint(tmin, tmax, size = (n, k), dtype = int)
    tu_output = tuple(map(tuple, output))
    li_output = []
    for m in tu_output:
        li_output.append(m)
    assert isinstance(li_output, list)
    assert isinstance(tu_output, tuple)
    return li_output


generateTuples(5,6,9,100)


# ### Q2

# #### a.

def computeSnippet(sample_list, ini, last):
    '''
    

    Parameters
    ----------
    sample_list : a list of tuple. 
        It has been given in this scenario: [(1, 3, 5), (0, 1, 2), (1, 9, 8)].
    ini : integer
        The first number in the tuple, refers to index 0.
    last : integer
        The last number in the tuple, refers to index -1.

    Returns
    -------
    res : a list of tuple
        The output is a compared result under the situation if the first element in the tuple are the same, pick 
        the tuple with the biggest last figure. If there's no common first element, then return the tuple.

    '''
    op = []
    for m in range(len(sample_list)):
        li = [sample_list[m]]
        for n in range(len(sample_list)):
            if (sample_list[m][ini] == sample_list[n][ini] 
                and sample_list[m][last] != sample_list[n][last]):
                li.append(sample_list[n])
            
        op.append(sorted(li, key=lambda dd: dd[last], reverse=True)[ini])
    res = list(set(op))
    return res


computeSnippet([(1, 3, 5), (0, 1, 2), (1, 9, 8)], 0, 2)


# #### b.

def updatedSnippet(sample_list, ini, last):
    '''
    

    Parameters
    ----------
    sample_list : a list of tuple. 
        It has been given in this scenario: [(1, 3, 5), (0, 1, 2), (1, 9, 8)].
    ini : integer
        The first number in the tuple, refers to index 0.
    last : integer
        The last number in the tuple, refers to index -1.

    Returns
    -------
    res : a list of tuple
        The output is a compared result under the situation if the first element in the tuple are the same, pick 
        the tuple with the biggest last figure. If there's no common first element, then return the tuple.

    '''
    tuple_list = []
    for m in range(len(sample_list)):
        temp_list = [sample_list[m]]
        for n in range(m+1, len(sample_list)):
            if (sample_list[m][ini] == sample_list[n][ini] and sample_list[m][last] != sample_list[n][last]):
                temp_list.append(sample_list[m])
                   
        tuple_list.append(sorted(temp_list, key=lambda dd: dd[last], reverse=True)[ini]) 
    res = list(set(tuple_list))
    return res


updatedSnippet([(1, 3, 5), (0, 1, 2), (1, 9, 8)], 0, 2)


# #### c.

def dicSnippet(sample_list, ini, last):
    '''
    

    Parameters
    ----------
    sample_list : a list of tuple. 
        It has been given in this scenario: [(1, 3, 5), (0, 1, 2), (1, 9, 8)].
    ini : integer
        The first number in the tuple, refers to index 0.
    last : integer
        The last number in the tuple, refers to index -1.

    Returns
    -------
    res : a list of tuple
        The output is a compared result under the situation if the first element in the tuple are the same, pick 
        the tuple with the biggest last figure. If there's no common first element, then return the tuple.

    '''
    sample_list.sort(key=lambda dd: dd[last], reverse=True)
    sample_list = set(sample_list)
    de = defaultdict(list)
    for t in sample_list:
        de[t[0]].append(t[1:])
    for key, value in de.items():
        de[key] = value[ini]
    se = de.items()
    se = list(se)
    return se


dicSnippet([(1, 3, 5), (0, 1, 2), (1, 9, 8)], 0, 2)


# #### d.

def computeSnippet(sample_list, ini, last):
    '''
    

    Parameters
    ----------
    sample_list : a list of tuple. 
        It has been given in this scenario: generateTuples(n, k, tmin, tmax).
    ini : integer
        The first number in the tuple, refers to index 0.
    last : integer
        The last number in the tuple, refers to index -1.

    Returns
    -------
    res : a list of tuple
        The output is a compared result under the situation if the first element in the tuple are the same, pick 
        the tuple with the biggest last figure. If there's no common first element, then return the tuple.

    '''
    op = []
    for m in range(len(sample_list)):
        li = [sample_list[m]]
        for n in range(len(sample_list)):
            if (sample_list[m][ini] == sample_list[n][ini] 
                and sample_list[m][last] != sample_list[n][last]):
                li.append(sample_list[n])
            
        op.append(sorted(li, key=lambda dd: dd[last], reverse=True)[ini])
    res = list(set(op))
    return res


computeSnippet(generateTuples(5,6,9,100),0, 5)


def updatedSnippet(sample_list, ini, last):
    '''
    

    Parameters
    ----------
    sample_list : a list of tuple. 
        It has been given in this scenario: generateTuples(n, k, tmin, tmax).
    ini : integer
        The first number in the tuple, refers to index 0.
    last : integer
        The last number in the tuple, refers to index -1.

    Returns
    -------
    res : a list of tuple
        The output is a compared result under the situation if the first element in the tuple are the same, pick 
        the tuple with the biggest last figure. If there's no common first element, then return the tuple.

    '''
    tuple_list = []
    for m in range(len(sample_list)):
        temp_list = [sample_list[m]]
        for n in range(m+1, len(sample_list)):
            if (sample_list[m][ini] == sample_list[n][ini] and sample_list[m][last] != sample_list[n][last]):
                temp_list.append(sample_list[m])
                   
        tuple_list.append(sorted(temp_list, key=lambda dd: dd[last], reverse=True)[ini]) 
    res = list(set(tuple_list))
    return res


updatedSnippet(generateTuples(5,6,9,100), 0, 5)


def dicSnippet(sample_list, ini, last):
    '''
    

    Parameters
    ----------
    sample_list : a list of tuple. 
        It has been given in this scenario: generateTuples(n, k, tmin, tmax).
    ini : integer
        The first number in the tuple, refers to index 0.
    last : integer
        The last number in the tuple, refers to index -1.

    Returns
    -------
    res : a list of tuple
        The output is a compared result under the situation if the first element in the tuple are the same, pick 
        the tuple with the biggest last figure. If there's no common first element, then return the tuple.

    '''
    sample_list.sort(key=lambda dd: dd[last], reverse=True)
    sample_list = set(sample_list)
    de = defaultdict(list)
    for t in sample_list:
        de[t[0]].append(t[1:])
    for key, value in de.items():
        de[key] = value[ini]
    se = de.items()
    se = list(se)
    return se


dicSnippet(generateTuples(5,6,9,100), 0, 5)

#  - each m is corresponding to three different functions and the output will be the mean time for running each function 1000 times.

n_mc = 1000
output = generateTuples(5,6,9,100)
ini = 0
m_list = []
for f in (computeSnippet(output, ini, 5), updatedSnippet(output, ini, 5), dicSnippet(output, ini, 5)):
    t = Timer("f", globals={"f": f})
    m = np.mean([t.timeit(1) for i in range(n_mc)]) 
    m = round(m * 1e6, 5)
    m_list.append(m)
print(m_list)

f_list = ["computeSnippet", "updatedSnippet", "dicSnippet"]
res = {}
for key in range(len(f_list)):
    res[f_list[key]] = m_list[key]
print(res)

dFrame = pd.DataFrame(res.items(), columns=['Function', 'Mean Time'])
dFrame

# +
cap = """
<b> Table 1.</b> <em> Timing comparisons for Tuple-comparison Functions.</em>
Calculating mean time from 1000 trials to see randomness in Marte Carlo.  
The method within dictionary is faster than the other two. 
"""

t1 = dFrame.to_html(index=False)
t1 = t1.rsplit('\n')
t1.insert(1, cap)
tab1 = ''
for i, line in enumerate(t1):
    tab1 += line
    if i < (len(t1) - 1):
        tab1 += '\n'
# -

display(HTML(tab1))

# ### Q3

#  - Import files from local directory

demo11_12 = pd.read_sas('/Users/Sylvia/Desktop/DEMO_G.XPT', encoding = 'utf-8')
demo13_14 = pd.read_sas('/Users/Sylvia/Desktop/DEMO_H.XPT', encoding = 'utf-8')
demo15_16 = pd.read_sas('/Users/Sylvia/Desktop/DEMO_I.XPT', encoding = 'utf-8')
demo17_18 = pd.read_sas('/Users/Sylvia/Desktop/DEMO_J.XPT', encoding = 'utf-8')

updateddemo11_12 = demo11_12[['SEQN','RIDAGEYR','RIDRETH3','DMDEDUC2','DMDMARTL','RIDSTATR','SDMVPSU','SDMVSTRA',
                             'WTMEC2YR','WTINT2YR']]
updateddemo13_14 = demo13_14[['SEQN','RIDAGEYR','RIDRETH3','DMDEDUC2','DMDMARTL','RIDSTATR','SDMVPSU','SDMVSTRA',
                             'WTMEC2YR','WTINT2YR']]
updateddemo15_16 = demo15_16[['SEQN','RIDAGEYR','RIDRETH3','DMDEDUC2','DMDMARTL','RIDSTATR','SDMVPSU','SDMVSTRA',
                             'WTMEC2YR','WTINT2YR']]
updateddemo17_18 = demo17_18[['SEQN','RIDAGEYR','RIDRETH3','DMDEDUC2','DMDMARTL','RIDSTATR','SDMVPSU','SDMVSTRA',
                             'WTMEC2YR','WTINT2YR']]

updateddemo11_12['cohort'] = '2011-2012'
updateddemo13_14['cohort'] = '2013-2014'
updateddemo15_16['cohort'] = '2015-2016'
updateddemo17_18['cohort'] = '2017-2018'

demo = pd.concat([updateddemo11_12, updateddemo13_14, updateddemo15_16, updateddemo17_18], axis = 0)
demo = demo.rename(columns ={'SEQN':'id','RIDAGEYR':'age','RIDRETH3':'race','DMDEDUC2':'education',
                            'DMDMARTL':'martial_status','RIDSTATR':'interview_status','SDMVPSU':'maskedvar-psu',
                            'SDMVSTRA':'maskedvar-stra','WTMEC2YR':'fullsample2yr_mec',
                            'WTINT2YR':'fullsample2yr_int'})
demo.head()

#  - Change non-categorical data into int

demo[['id','age','race']] = demo[['id','age','race']].astype(int)
demo.head()

#  - Change categorical data as category

demo[['education', 'martial_status', 'interview_status']] = demo[['education', 'martial_status', 'interview_status']].astype('category')
demo.head()

#  - Keep range data to 2 decimal places and change into categorical data

demo[['fullsample2yr_mec','fullsample2yr_int']] = demo[['fullsample2yr_mec','fullsample2yr_int']].round(2)
demo['fullsample2yr_mec'] = pd.to_numeric(demo['fullsample2yr_mec'], errors='coerce')
demo['maskedvar-psu'] = pd.to_numeric(demo['maskedvar-psu'], errors='coerce')
demo['maskedvar-stra'] = pd.to_numeric(demo['maskedvar-stra'], errors='coerce')
demo['cohort'] = demo['cohort'].astype('category')
demo.dtypes

# #### Save to pickle

pickle_out_demo = open("demo.pickle","wb")
pickle.dump(demo, pickle_out_demo)
pickle_out_demo.close()

pickle_in_demo = open("demo.pickle","rb")
example_demo = pickle.load(pickle_in_demo)
print(example_demo)

#  - Import Oral dataset from local directory

oral11_12 = pd.read_sas('/Users/Sylvia/Desktop/OHXDEN_G.XPT', encoding = 'utf-8')
oral13_14 = pd.read_sas('/Users/Sylvia/Desktop/OHXDEN_H.XPT', encoding = 'utf-8')
oral15_16 = pd.read_sas('/Users/Sylvia/Desktop/OHXDEN_I.XPT', encoding = 'utf-8')
oral17_18 = pd.read_sas('/Users/Sylvia/Desktop/OHXDEN_J.XPT', encoding = 'utf-8')

oral11_12CTC = oral11_12.loc[:, oral11_12.columns.str.endswith('TC')]

updatedoral11_12 = oral11_12[['SEQN','OHDDESTS']]
updatedoral11_12 = pd.concat([updatedoral11_12, oral11_12CTC], axis = 1)

#  - Extract columns with keyword "TC" for both TC and CTC. However, there exist data ended by "RTC" which is needed to be removed

oral13_14CTC = oral13_14.loc[:, oral13_14.columns.str.endswith('TC')]
updatedoral13_14 = oral13_14[['SEQN','OHDDESTS']]
updatedoral13_14 = pd.concat([updatedoral13_14, oral13_14CTC], axis = 1)
oral15_16CTC = oral15_16.loc[:, oral15_16.columns.str.endswith('TC')]
oral15_16CTC = oral15_16CTC.loc[:, [not o.endswith('RTC') for o in oral15_16CTC.columns]]
updatedoral15_16 = oral15_16[['SEQN','OHDDESTS']]
updatedoral15_16 = pd.concat([updatedoral15_16, oral15_16CTC], axis = 1)
oral17_18CTC = oral17_18.loc[:, oral17_18.columns.str.endswith('TC')]
oral17_18CTC = oral17_18CTC.loc[:, [not o.endswith('RTC') for o in oral17_18CTC.columns]]
updatedoral17_18 = oral17_18[['SEQN','OHDDESTS']]
updatedoral17_18 = pd.concat([updatedoral17_18, oral17_18CTC], axis = 1)

updatedoral11_12['cohort'] = '2011-2012'
updatedoral13_14['cohort'] = '2013-2014'
updatedoral15_16['cohort'] = '2015-2016'
updatedoral17_18['cohort'] = '2017-2018'

oral = pd.concat([updatedoral11_12, updatedoral13_14, updatedoral15_16, updatedoral17_18], axis = 0)

#  - Change int data into int 

oral = oral.rename(columns = {'SEQN':'id', 'OHDDESTS':'dent_code'})
oral[['id','dent_code']] = oral[['id','dent_code']].astype(int)

#  - The rest will be category

oral.iloc[:,2:] = oral.iloc[:,2:].astype('category')
oral.dtypes

# #### Save to pickle

pickle_out_oral = open("oral.pickle","wb")
pickle.dump(oral, pickle_out_oral)
pickle_out_oral.close()

pickle_in_oral = open("oral.pickle","rb")
example_oral = pickle.load(pickle_in_oral)
print(example_oral)

# #### d. Number of cases

#Number of cases of Demo datasets
demo.shape[0]

demo.describe()

#Number of cases od Oral datasets
oral.shape[0]

oral.describe()
