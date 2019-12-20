# -*- coding: utf-8 -*-
"""
@Author: Hai
The script works on Python >= 3.6 and requires pandas, matplotlib, numpy and scipy to work.
The current verison of scripts will give you some warns, you may ignore them.
The scipt works with the alonged script growth2fig_manager.py.
Please use the manager and it is not necessary to modify this script for working. 
"""

import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import linregress, sem
from lmfit import minimize, Parameters

plt.rc('axes', axisbelow=True)

def get_valid_filename(s):
    """
    Return the given string converted to a string that can be used for a clean
    filename. Remove leading and trailing spaces; convert other spaces to
    underscores; and remove anything that is not an alphanumeric, dash,
    underscore, or dot.
    Adapted from Django Framework, utils/text.py
    """
    s = str(s).splitlines()[0]
    s = s.strip().replace(' ', '_')
    s = re.sub(r'(?u)[^-\w.]', '', s)
    s = re.sub('mathit','',s)
    return re.sub('Delta_', 'D', s)
    
def cal(file_name, inc_OD):   # OD calibration
    data = pd.read_excel(file_name, sheet_name='Sheet1', index_col=0).dropna()
    data.iloc[:,1:] = data.iloc[:,1:] - data.iloc[:,1:5].values.min()
    data.iloc[:,1:] = data.iloc[:,1:] / 0.23 + inc_OD  # for infinate
    time = data.index.values.astype(np.float) / 3600  # in hours
    data.index = time
    return data

def grfunc(params, x, y=None):
    n0 = params['n0']
    mu = params['mu']  # in hour

    model = n0 * np.exp(mu * x)
    if y is None:
        return model
    return model - y

def plot(data, MATS, WIN, TH, Figure_Type, Yscale, Xmax, Ymax, cmap, linestyles, savefig='no', errb=1):   
    for mat in MATS.loc[2]:
        slope = np.array([])
        timePoint = np.array([])
        biomassYield = np.array([])
        for id in mat:
            v = data[id]
            if np.amax(v) < TH:
                slope = np.append(slope, 0.001)   # 0.001 is just a number small enough
                timePoint = np.append(timePoint, np.inf)
                biomassYield = np.append(biomassYield, np.amax(v))
                continue
            else:
                rate_tmp = np.array([])
                for tp in v[0:v.index.max()-WIN].index:
                    # P = linregress(v.loc[tp:tp+WIN].index.values, np.log(v.loc[tp:tp+WIN]))
                    # rate_tmp = np.append(rate_tmp, P.slope) # more: intercept, rvalue, pvalue, stderr
                    t = v.loc[tp:tp+WIN].index.to_numpy()
                    od = v.loc[tp:tp+WIN].to_numpy()
                    params = Parameters()
                    params.add('n0', value=0.001)
                    params.add('mu', value=0.001)
                    fitres = minimize(grfunc, params, args=(t,), kws={'y': od})
                    rate_tmp = np.append(rate_tmp, fitres.params['mu'].value)
                slope = np.append(slope, np.nanmax(rate_tmp))
                timePoint = np.append(timePoint,
                                      v.index.values[np.nanargmax(rate_tmp)])
                biomassYield = np.append(biomassYield, np.amax(v))
        
        doubling_time = np.around(np.log(2) / slope, 1)
        doubling_time_mean = np.around(np.nanmean(doubling_time), 1)
        doubling_time_sd = np.around(np.nanstd(doubling_time), 1)
        # doubling_time_sem = np.around(sem(doubling_time), 1)
        time_start_mean = np.around(np.nanmean(timePoint), 1)
        time_start_sd = np.around(np.nanstd(timePoint), 1)
        # time_start_sem = np.around(sem(timePoint), 1)
        biomassYield_mean = np.around(np.nanmean(biomassYield), 2)
        biomassYield_sd = np.around(np.nanstd(biomassYield), 1)
        # biomassYield_sem = np.around(sem(biomassYield), 1)
        
        if doubling_time_mean < 150:
            MATS.loc[3][MATS.loc[2].index(mat)] = MATS.loc[3][MATS.loc[2].index(mat)] \
                                                + ': ' + str(doubling_time_mean) \
                                                + '(' + str(doubling_time_sd) \
                                                + '), ' + str(time_start_mean) \
                                                + '(' + str(time_start_sd) + '), ' \
                                                + str(biomassYield_mean) + '(' \
                                                + str(biomassYield_sd) + ')'
        else:
            MATS.loc[3][MATS.loc[2].index(mat)] = MATS.loc[3][MATS.loc[2].index(mat)] + ': NG'
    
    # ploting all/mean/errorbar
    fig, ax = plt.subplots(figsize=(12,10))
    lines = []
    line_num = []
    
    for mat in MATS.loc[2]:
        i = MATS.loc[2].index(mat)
        if Figure_Type == 'mean':
            line_num.append(len(lines))
            lines += ax.plot(data.index, data[mat].mean(axis=1),
                             linestyle=linestyles[i], color=cmap[i], linewidth=3)    # label=MATS[3][i] 
        elif Figure_Type =='all':
            line_num.append(len(lines))
            lines += ax.plot(data.index, data[mat],label=MATS[3][i],
                             linestyle=linestyles[i], color=cmap[i], linewidth=3)
        elif Figure_Type == 'errorbar':
            line_num.append(len(lines))
            lines += ax.errorbar(data.index, data[mat].mean(axis=1),
                                 yerr=data[mat].std(axis=1), capsize=4, barsabove=True, errorevery=errb,
                                 linestyle=linestyles[i], color=cmap[i], linewidth=3,)
        
    ax.grid(which='major', axis='x', linewidth=1, linestyle='-', color='0.75')
    ax.grid(which='major', axis='y', linewidth=1, linestyle='--', color='0.75')
    ax.tick_params(which='both',direction='in', bottom=True, right=False, 
                   top=False, left=True, labelsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yscale(Yscale)

    if Yscale == 'log':
        ax.set_ylim(top=Ymax)
    elif Yscale == 'linear':
        ax.set_ylim(bottom=0,top=Ymax)
    
    ax.set_title(MATS[1], fontsize = 24)
    ax.set_ylabel('OD600', fontsize = 18)
    ax.set_xlabel('Time (h)', fontsize = 18)
    ax.set_xlim(left=0, right=Xmax)
    ax.legend([lines[i] for i in line_num], MATS[3][:], fontsize=16)
    
    if savefig != 'no':
        title = get_valid_filename(MATS[1])
        fig.savefig('{}_{}.{}'.format(Figure_Type, title,savefig),dpi=300)

def viewPlate(data, nwells=96):
    """ Please confirm the type of plate,
    96-well or others.
    """
    if nwells == 96:
        nrows = 8
        ncols = 12
    elif nwells == 48:
        nrows = 6
        ncols = 8
    row = ['A','B','C','D','E','F','G','H']
    col = list(range(1,13))

    fig, axes = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True)
    fig.set_size_inches(15, 10)

    for i in range(nrows):
        for j in range(ncols):
            well = row[i] + str(col[j])
            if well in data.columns.values.tolist():
                axes[i,j].plot(data.index, data.loc[:,well], linewidth=3)

    plt.show()

def wellPlot(Wells, data):
    """Please enter wells in list, e.g. ['A1','A2',]"""
    ID = data.columns.values.tolist()
    plt.figure(figsize=(12, 10))
    if np.all(np.isin(Wells, ID)):
        plt.plot(data.index, data.loc[:,Wells], linewidth=3)
        plt.legend(Wells)
        plt.show()
    else:
        print("Please enter valid wells, e.g. ['A1','A2','A3']")