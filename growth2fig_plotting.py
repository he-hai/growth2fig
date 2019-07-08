# -*- coding: utf-8 -*-
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import linregress, sem

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
    
def cal(file_name, inc_OD):   # calculation
    data = pd.read_excel(file_name, sheet_name='Sheet1', index_col=0).dropna()
    data.iloc[:,1:] = data.iloc[:,1:] - data.iloc[:,1:].values.min()
    data.iloc[:,1:] = data.iloc[:,1:] / 0.23 + inc_OD  # for infinate
    time = data.index.values.astype(np.float) / 3600  # in hours
    data.index = time
    return data
    
def plot(data, MATS, WIN, TH, Figure_Type, Yscale, Xmax, Ymax, cmap, linestyles, savefig='no'):   
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
                for tp in v.index:
                    P = linregress(v.loc[tp:tp+WIN].index.values, np.log(v.loc[tp:tp+WIN]))
                    rate_tmp = np.append(rate_tmp, P.slope) # more: intercept, rvalue, pvalue, stderr
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

    # ploting all/mean
    fig, ax = plt.subplots()
    lines = []
    line_num = []
    
    for mat in MATS.loc[2]:
        i = MATS.loc[2].index(mat)
        if Figure_Type == 'mean':
            line_num.append(len(lines))
            lines += ax.plot(data.index, data[mat].mean(axis=1),
                             linestyle=linestyles[i], color=cmap[i], linewidth=2)        # label=MATS[3][i] 
        elif Figure_Type =='all':
            line_num.append(len(lines))
            lines += ax.plot(data.index, data[mat],label=MATS[3][i],
                             linestyle=linestyles[i], color=cmap[i], linewidth=2)
        
    ax.grid(which='major', axis='both', linewidth=1, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='y', linewidth=0.75, linestyle='--', color='0.75')
    ax.tick_params(which='both',direction='in', bottom=True, right=False, 
                   top=False, left=True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yscale(Yscale)
    if Yscale == 'log':
        ax.set_ylim(top=Ymax)
    elif Yscale == 'linear':
        ax.set_ylim(bottom=0,top=Ymax)
    plt.setp(ax, title=MATS[1], ylabel='OD600', xlabel='Time (h)',
         xlim=(0, Xmax))
    ax.legend([lines[i] for i in line_num], MATS[3][:])
    if savefig != 'no':
        title = get_valid_filename(MATS[1])
        fig.savefig('{}_{}.{}'.format(Figure_Type, title,savefig),dpi=300)