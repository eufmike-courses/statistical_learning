# %%
import os, sys
import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.tools as stat
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %% 
# load data
path = '/Users/michaelshih/Documents/code/education/statistical_learining/'
subfolder = 'resource'
filename = 'Smarket.csv'
filedir = os.path.join(path, subfolder, filename)
print(filedir)

data = pd.read_csv(filedir, index_col = 0, sep = ";")
data = pd.DataFrame(data)
display(data)

