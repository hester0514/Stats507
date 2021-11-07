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

# ## STATS507 - Problem Set 1

# Xuechun Wang<br>
#
# UMICHID:24107190

import numpy as np
import pandas as pd
import math
import timeit
from random import randrange
from collections import defaultdict
import scipy.stats as stats
import warnings
from prettytable import PrettyTable
import string
from tabulate import tabulate
from IPython.core.display import display, HTML
from scipy.stats import norm, binom, beta
from warnings import warn


# ## Question 0 - Markdown warmup

# This is *question 0* for [problem set 1](https://jbhender.github.io/Stats507/F21/ps/ps1.html) of [Stats 507](https://jbhender.github.io/Stats507/F21/).<br>
# >### Question 0 is about Markdown.<br>

# The next question is about **Fibonnaci sequence**, $F_{n}=F_{n-1}+F_{n-2}$. In part **a** we will define a Python function ``fib_rec()``.<br>
# Below is a...<br>
# <br>

# ### Level 3 Header<br>
# Next, we can make a bullet list:<br>
# - Item 1<br>
#      - detail 1
#      - detail 2
# - Item 2<br>
# Finally, we can make an enumerated list:<br>
# - a. Item 1
# - b. Item 2
# - c. Item 3
#      

# ## Q1 - Fibonacci Sequence

# ### a. fib_rec()

def fib_rec(n, a = 0, b = 1):
    '''
    

    Parameters
    ----------
    n : integer
        Number of input.
    a : integer, optional
        The initial value when n is 0. The default is 0.
    b : integer, optional
        The initial value when n is 1. The default is 1.

    Returns
    -------
    TYPE int
        Return Fibonacci Sequence number.

    '''
    F_0 = a
    F_1 = b
    if n == 0:
        return F_0
    elif n == 1:
        return F_1
    elif n < 0:
        return "n cannot be negative"
    else:
        return fib_rec(n-1) + fib_rec(n-2)


# ### b. fib_for()

def fib_for(n):
    '''
    

    Parameters
    ----------
    n : integer
        Number of input.

    Returns
    -------
    TYPE int
        Return Fibonacci Sequence number.

    '''
    if n == 0:
        return 0
    elif n == 1:
        return 1
    elif n < 0:
        return "n cannot be negative"
    else: 
        firstInitial = 0
        secondInitial = 1
        count  = 0
        for count in range(n-1):
            sumOfValues = firstInitial + secondInitial
            firstInitial = secondInitial
            secondInitial = sumOfValues
            count += 1
        return sumOfValues


# ### c. fib_whl

def fib_whl(n):
    '''
    

    Parameters
    ----------
    n : int
        number of input.

    Returns
    -------
    TYPE int
        Return Fibonacci Sequence number.

    '''
    if n == 0:
        return 0
    elif n == 1:
        return 1
    elif n < 0:
        return "n cannot be negative"
    while n >= 2:
        summation = fib_whl(n-1) + fib_whl(n-2)
        return summation 


# ### d. fib_rnd()

def fib_rnd(n):
    '''
    

    Parameters
    ----------
    n : int
        number of input.

    Returns
    -------
    TYPE int
        Return Fibonacci Sequence number.

    '''
    if n < 0:
        return n
    else:
        pho = (1+math.sqrt(5))/2
        fn = (pho ** n)/math.sqrt(5)
        return math.ceil(fn)


# ### e. fib_flr()

def fib_flr(n):
    '''
    

    Parameters
    ----------
    n : int
        number of input.

    Returns
    -------
    TYPE int
        Return Fibonacci Sequence number.

    '''
    if n < 0:
        return n
    else:
        pho = (1+math.sqrt(5))/2
        fn = (pho ** n)/math.sqrt(5) + 1/2
        return math.floor(fn)


# ### f. execution time

count = [7,11,13]
time_lst = defaultdict(list)
for f in [fib_rec, fib_for, fib_whl, fib_rnd, fib_flr]:
    for var in count:
        t = timeit.Timer("f(var)",globals = {'f': f, 'var':var})
        tm = t.repeat(repeat=100, number=50)
        time_lst['Function'].append(f.__name__)
        time_lst['median, s'].append(np.median(tm))
time_lst = pd.DataFrame(time_lst)
for c, d in zip(time_lst.columns, time_lst.dtypes):
    if d == np.dtype('float64'):
        time_lst[c] = time_lst[c].map(lambda x: '%5.3f' % x)
time_lst


# ### Q2 - Pascal’s Triangle

# ### a. SpecificRow

def specifiedRow(n):
    rowNum = [1]
    for k in range(n):
        rowNum.append(round(rowNum[k] * (n-k) / (k+1)))
    return print(str(rowNum).strip('[]'))


# ### b. Print Triangle

def picTriangle(n):
    for i in range(n+1):
        row = 1
        for j in range(n-i+1):
            print(format(" ","<2"), end ="")        
        print(" ",row, end = "   ")
        for j in range(1,i+1):
            row = int(row*(i+1-j)/j)
            print(format(row, "<3"),end =" ")
        print()


specifiedRow(11)

picTriangle(11)


# ### Q3 - Statistics 101

# ### a. Normal Theory

# +
# def ci_format():
#     ci_format = {'Lower CI': None, 'Upper CI': None, 'Est': None, 'cLevel': None}
#     return ci_format
# -

def computeEstimate(data, alpha):
    #ci_format = ciformat()
    randomValues = np.array(data)
    mean, sigma = np.mean(randomValues), np.std(randomValues)
    se = sigma/math.sqrt(len(randomValues))
    zValue = stats.norm.ppf(1 - alpha, 0 ,1)
    cLevel = (1 - alpha)*100
    ci_lo = mean - zValue * se
    ci_hi = mean + zValue * se
    result = {'Methods': 'Normal Theory','Lower CI': "%.3f" %ci_lo, 'Upper CI': "%.3f" %ci_hi, 'Est': "%.3f" %mean, 'Level': cLevel}
    #if ci_format == None:
    #    return result
    #else:
    #return '{Est}[{cLevel}%CI:({ci_lo},{ci_hi})]'.format_map(result)
    return result


computeEstimate([1,2,3,4,5],0.05)


# ### b. Confidence Level

def computeConfiInt(n, x, alpha, methods):
    randomValues = np.asarray(n)
    zValue = stats.norm.ppf(1 - alpha/2)
    pHat = x/len(randomValues)
    cLevel = (1 - alpha)*100
    if methods == 'Normal Approximation':
        if min((len(randomValues))*pHat, (len(randomValues))*(1-pHat))>12:
            ci_lo = pHat - zValue * math.sqrt(pHat*(1-pHat)/len(randomValues))
            ci_hi = pHat + zValue * math.sqrt(pHat*(1-pHat)/len(randomValues))
            result = {'Methods': methods, 'Lower CI': "%.3f" %ci_lo, 'Upper CI': "%.3f" %ci_hi, 'Level': cLevel}
        else:
            return warnings.warn("The np value or n(1-p) value is below 12")
    
    elif methods == 'Clopper-Pearson':
        ci_lo = stats.beta.ppf(alpha/2, x, len(randomValues) - x + 1)
        ci_hi = stats.beta.ppf(1 - alpha/2, x + 1, len(randomValues) - x)
        result = {'Methods': methods,'Lower CI': "%.3f" %ci_lo, 'Upper CI': "%.3f" %ci_hi, 'Level': cLevel}
        
    elif methods == 'Jeffrey’s interval':
        ci_lo = max(stats.beta.ppf(alpha/2, x + 0.5, len(randomValues) - x + 0.5),0)
        ci_hi = min(stats.beta.ppf(1 - alpha/2, x + 0.5, len(randomValues) - x + 0.5),1)
        result = {'Methods': methods, 'Lower CI': "%.3f" %ci_lo, 'Upper CI': "%.3f" %ci_hi, 'Level': cLevel}
        
    elif methods == 'Agresti-Coull interval':
        nbar = len(randomValues) + zValue ** 2
        pbar = (x + (zValue**2)/2)/nbar
        ci_lo = pbar - zValue*math.sqrt(pbar*(1-pbar)/nbar)
        ci_hi = pbar + zValue*math.sqrt(pbar*(1-pbar)/nbar)
        result = {'Methods': methods, 'Lower CI': "%.3f" %ci_lo, 'Upper CI': "%.3f" %ci_hi, 'Level': cLevel}
        
    else:
        return 'Method does not exist.'
    
    return result


computeConfiInt([1,2,3,4,5,6,7,8,9,10,11,12,13],2, 0.05, 'Agresti-Coull interval')


def ci_prop(
    x,
    level=0.95,
    str_fmt="{mean:.2f} [{level:.0f}%: ({lwr:.2f}, {upr:.2f})]",
    method="Normal"
):
    try:
        x = np.asarray(x)  # or np.array() as instructed.
    except TypeError:
        print("Could not convert x to type ndarray.")

    # check that x is bool or 0/1
    if x.dtype is np.dtype('bool'):
        pass
    elif not np.logical_or(x == 0, x == 1).all():
        raise TypeError("x should be dtype('bool') or all 0's and 1's.")

    # check method
    assert method in ["Normal", "CP", "Jeffrey", "AC"]

    # determine the length
    n = x.size

    # compute estimate
    if method == 'AC':
        z = norm.ppf(1 - (1 - level) / 2)
        n = (n + z ** 2)
        est = (np.sum(x) + z ** 2 / 2) / n
    else:
        est = np.mean(x)

    # warn for small sample size with "Normal" method
    if method == 'Normal' and (n * min(est, 1 - est)) < 12:
        warn(Warning(
            "Normal approximation may be incorrect for n * min(p, 1-p) < 12."
        ))

    # compute bounds for Normal and AC methods
    if method in ['Normal', 'AC']:
        se = np.sqrt(est * (1 - est) / n)
        z = norm.ppf(1 - (1 - level) / 2)
        lwr, upr = est - z * se, est + z * se

    # compute bounds for CP method
    if method == 'CP':
        alpha = 1 - level
        s = np.sum(x)
        lwr = beta.ppf(alpha / 2, s, n - s + 1)
        upr = beta.ppf(1 - alpha / 2, s + 1, n - s)

    # compute bounds for Jeffrey method
    if method == 'Jeffrey':
        alpha = 1 - level
        s = np.sum(x)
        lwr = beta.ppf(alpha / 2, s + 0.5, n - s + 0.5)
        upr = beta.ppf(1 - alpha / 2, s + 0.5, n - s + 0.5)

    # prepare return values
    out = {"mean": est, "level": 100 * level, "lwr": lwr, "upr": upr}
    if str_fmt is None:
        return(out)
    else:
        return(str_fmt.format_map(out))


ci_prop(
    [1,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,1,1,1],
    level=0.95,
    str_fmt="{mean:.2f} [{level:.0f}%: ({lwr:.2f}, {upr:.2f})]",
    method="Normal"
)

# ### c. Table Format

zeros = np.zeros(48)
ones = np.ones(42)
arr = np.concatenate((zeros, ones), axis=0)

# #### Data Arrangement

alphaVal = [0.10, 0.05, 0.01]
confiList = ['Normal Approximation','Clopper-Pearson','Jeffrey’s interval','Agresti-Coull interval']
emptyList = []
for val in alphaVal:
    result_a = computeEstimate(arr, val)
    emptyList.append(result_a)
    for i in confiList:
        result_b = computeConfiInt(arr, 42, val, i)
        emptyList.append(result_b)
dFrame = pd.DataFrame(emptyList)
dFrame['Confi Inter'] = '(' + dFrame['Lower CI'] + ' , ' + dFrame['Upper CI'] + ')'
dFrame['Confi Level'] = dFrame['Level'].astype(str) + '%'
dFrame = dFrame.sort_values(by = 'Methods')
dFrame['Width'] = dFrame['Upper CI'].astype(float) - dFrame['Lower CI'].astype(float)
dFrame = dFrame.sort_values(by = ['Methods','Width'])
dFrame_extracted = dFrame[['Confi Level', 'Confi Inter', 'Width']]
dFrame_extracted

# #### Average width

dFrame_pivot = pd.pivot_table(dFrame, index = ['Methods','Confi Level'], values = 'Width', aggfunc = np.mean)
dFrame_pivot

# #### Table Display

table = tabulate(dFrame_extracted, headers = 'keys',floatfmt= (".2f",".2f"))
print(table)
