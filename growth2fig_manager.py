#%%
# -*- coding: utf-8 -*-
"""
@Author: Hai
The script works on Python 3, you may use VS code (higher than 1.35) in the interactive mode 
or iPython IDE, such as Spyder.
The script requires pandas, matplotlib, numpy, scipy and the alonged script 'growth2fig_plotting.py' to work.
The current verison of scripts will give you some warns, you may ignore them.

Your data should be prepared in excel first, in the format like the test data set.
It has wells in columns, and time in rows. Data from Biotech machine is perfect fine. 
Please transpose data from Tecan.
"""

# from IPython import get_ipython
# get_ipython().magic('reset -sf')  # equal to clear in matlab
# get_ipython().magic('matplotlib qt')

import pandas as pd
from matplotlib import pyplot as plt
import growth2fig_plotting as g2plot

plt.close('all')  # close all graph windows
experiment = pd.DataFrame() # DO NOT modify this.

DataFileName = 'testingData.xlsx'  # data need to be prepared first in excel, and put it into the same folder.
inc_OD = 0.02  # the starting OD in cuvette unit
# Figure_Type = 'all'
# Figure_Type = 'mean'
Figure_Type = 'errorbar'
errb = 5  # plot error bars every 5 data points
Yscale = 'log'
# Yscale = 'linear'
Xmax = None  # None or setting with number, in hours
Ymax = None  # None or setting with number, OD
WIN = 5  # in hours, period in which growth rate is measured
TH = 0.05  # Maximum OD below this value is NO GROWTH
savefig = 'no'   # eps, png, or no to save graphs as .eps or .png files named by graph titles.

# set titles, 'Title'
experiment.loc[1,1] = 'title 1'
experiment.loc[1,2] = 'title 2'
experiment.loc[1,3] = 'title 3'
experiment.loc[1,4] = 'title 4'
experiment.loc[1,5] = 'title 5'
experiment.loc[1,6] = 'title 6'
experiment.loc[1,7] = 'title 7'
experiment.loc[1,8] = 'title 8'
experiment.loc[1,9] = 'title 9'
experiment.loc[1,10] = 'title 10'

# set replicates; ['A1','A2']
experiment.loc[2,1] = [['A1','A2'], ['A3','A4'],['A5','A6']]
experiment.loc[2,2] = [['B1','B2'], ['B3','B4'],['B5','B6']]
experiment.loc[2,3] = [['C1','C2'], ['C3','C4'],['C5','C6']]
experiment.loc[2,4] = [['D1','D2'], ['D3','D4'],['D5','D6']]
experiment.loc[2,5] = [['E1','E2'], ['E3','E4'],['E5','E6']]
experiment.loc[2,6] = [['F1','F2'], ['F3','F4'],['F5','F6']]
experiment.loc[2,7] = [['G1','G2'], ['G3', 'G4'], ['G5', 'G6']]
experiment.loc[2,8] = [['A7','A8'], ['B7', 'B8'], ['C7', 'C8'], ['D7', 'D8'], ['E7', 'E8'], ['F7', 'F8']]
experiment.loc[2,9] = [['A9','A10'], ['B9', 'B10'], ['C9', 'C10'], ['D9', 'D10'], ['E9', 'E10'], ['F9', 'F10']]
experiment.loc[2,10] = [['A11','A12'], ['B11', 'B12'], ['C11', 'C12'], ['D11', 'D12'], ['E11', 'E12'], ['F11', 'F12']]

# set condistions
experiment.loc[3,1] = ['condition 1', 'condition 2', 'condition 3']
experiment.loc[3,2] = ['condition 1', 'condition 2', 'condition 3']
experiment.loc[3,3] = ['condition 1', 'condition 2', 'condition 3']
experiment.loc[3,4] = ['condition 1', 'condition 2', 'condition 3']
experiment.loc[3,5] = ['condition 1', 'condition 2', 'condition 3']
experiment.loc[3,6] = ['condition 1', 'condition 2', 'condition 3']
experiment.loc[3,7] = ['condition 1', 'condition 2', 'condition 3']
experiment.loc[3,8] = ['strain 1', 'strain 2', 'strain 3', 'strain 4', 'strain 5', 'strain 6']
experiment.loc[3,9] = ['strain 1', 'strain 2', 'strain 3', 'strain 4', 'strain 5', 'strain 6']
experiment.loc[3,10] = ['strain 1', 'strain 2', 'strain 3', 'strain 4', 'strain 5', 'strain 6']

# set linestyles, preset 10 solid lines for all here
linestyles = ['-','-','-','-','-','-','-','-','-','-']

# set colors, preset 10 colors here
cmap = ['k', 'b', 'r','g','c','m','y','saddlebrown','orange','olive']

# calculations, this calculates doubling time ...
DATA = g2plot.cal(DataFileName, inc_OD)

# ploting
for column in experiment:
    g2plot.plot(DATA, experiment[column], WIN, TH, Figure_Type, Yscale, 
                Xmax, Ymax, cmap, linestyles, savefig, errb)
plt.show()

#%% 
# view the whole plate, change nwells if not 96-well plate
# Uncomment it if needed.
# g2plot.viewPlate(DATA, nwells=96)

#%% 
# plot a list of wells
# Uncomment it if needed.
# Wells = ['A1', 'B1']  # wells to be plotted
# g2plot.wellPlot(Wells, DATA)
