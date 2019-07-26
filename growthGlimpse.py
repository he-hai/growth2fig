#%%
# -*- coding: utf-8 -*-
"""
@Author: Hai
The script works on Python >= 3.6 and requires pandas, matplotlib, numpy and scipy to work.
The current verison of scripts will give you some warns, you may ignore them.
The scipt works with the alonged script growth2fig_manager.py.
Please use the manager and it is not necessary to modify this script for working. 
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import loadmat
"""
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
"""
mat = loadmat('DATA.mat')

ID_tmp = mat['ID']
ID = []
for i in range(ID_tmp.size):
    if ID_tmp[i,0].size > 0:
        ID.append(ID_tmp[i,0][0])
    else:
        ID.append(i)

data = pd.DataFrame(mat['DATA'],index=ID).dropna()

# inc_OD = float(input('Enter the inoculated OD (default 0.02): ') or '0.02')
inc_OD = 0.02

time = data.iloc[0] / 3600
data.iloc[2:,:] = data.iloc[2:,:] - data.iloc[2:10,:].values.min()
data.iloc[2:,:] = data.iloc[2:,:] / 0.23 + inc_OD

# Wells = input('Enter wells (separate by a blank): ').split()
def wellPlot(Wells):
    """Please enter wells in list, e.g. ['A1','A2',]"""
    # Wells = Wells.split()
    # plt.close('all')
    plt.figure()
    if np.all(np.isin(Wells, ID)):
        plt.plot(time,data.loc[Wells].T)
        plt.legend(Wells)
        plt.show()
    else:
        print("Please enter valid wells, e.g. ['A1','A2','A3']")
# %%
# wellPlot(['A2'])

# %%
# overview of the whole plate

nwells = 96
# nwells = 48
if nwells == 96:
    nrows = 8
    ncols = 12
elif nwells == 48:
    nrow = 6
    ncols = 8
row = ['A','B','C','D','E','F','G','H']
col = list(range(1,13))

fig, axes = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True)

for i in range(nrows):
    for j in range(ncols):
        well = row[i] + str(col[j])
        if well in data.index:
            axes[i,j].plot(time,data.loc[well].T)

plt.show()

# %%
