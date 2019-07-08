#%%
# -*- coding: utf-8 -*-
import pandas as pd
from matplotlib import pyplot as plt
import growth2fig_plotting as g2plot

plt.close('all')
experiment = pd.DataFrame()

DataFileName = '20190117_ltaE sdaA promoter.xlsx'  # data need to be prepared first in excel
inc_OD = 0.02  # the starting OD in cuvette unit
Figure_Type = 'all'
# Figure_Type = 'mean'
#Yscale = 'log'
Yscale = 'linear'
Xmax = 120  # None or setting with number
Ymax = 2.8  # None or setting with number
WIN = 5  # in hours, period in which growth rate is measured
TH = 0.05  # 
savefig = 'no'   # eps, png, or no

# set titles 'Keio $\mathit{\Delta purU}$, 10mM glucose'
experiment.loc[1,1] = 'fC1SAUX ASS-ltaE-CgadhA \n10mM glc, 10mM Gly, 0.5uM Mn'
experiment.loc[1,2] = 'fC1Sac PltaES ASS-CgadhA \n10mM glc, 10mM Gly, 0.5uM Mn'
experiment.loc[1,3] = 'fC1Sac ASS-ltaE-CgadhA \n10mM glc, 10mM Gly, 0.5uM Mn'
experiment.loc[1,4] = 'fC1Sac ASS-CgadhA-ltaE \n10mM glc, 10mM Gly, 0.5uM Mn'
experiment.loc[1,5] = '$\mathit{\Delta frm \Delta glyA \Delta serA}$ ASS-ltaE-CgadhA \n10mM glc, 10mM Gly, 0.5uM Mn'
experiment.loc[1,6] = 'fC1Sac sdaA P_wt \n0.5uM Mn'
experiment.loc[1,7] = 'fC1Sac PsdaAS \n0.5uM Mn'
experiment.loc[1,8] = 'fC1Sac ASS-sdaA \n0.5uM Mn'

# set replicates
experiment.loc[2,1] = [['A1','A2'], ['A3','A4'],['A5','A6'],['A7','A8'],['A9','A10'],['A11','A12']]
experiment.loc[2,2] = [['B1','B2'], ['B3','B4'],['B5','B6'],['B7','B8'],['B9','B10'],['B11','B12']]
experiment.loc[2,3] = [['C1','C2'], ['C3','C4'],['C5','C6'],['C7','C8'],['C9','C10'],['C11','C12']]
experiment.loc[2,4] = [['D1','D2'], ['D3','D4'],['D5','D6'],['D7','D8'],['D9','D10'],['D11','D12']]
experiment.loc[2,5] = [['E1','E2'], ['E3','E4'],['E5','E6'],['E7','E8'],['E9','E10'],['E11','E12']]
experiment.loc[2,6] = [['F1','F2'], ['F3','F4'],['F5','F6'],['F7','F8'],['F9','F10'],['F11','F12']]
experiment.loc[2,7] = [['G1','G2'], ['G3', 'G4'], ['G5', 'G6'], ['G7', 'G8'], ['G9', 'G10'], ['G11', 'G12']]
experiment.loc[2,8] = [['H1','H2'], ['H3', 'H4'], ['H5', 'H6'], ['H7', 'H8'], ['H9', 'H10'], ['H11', 'H12']]

# set condistions
experiment.loc[3,1] = ['1M MeOH', '0.75M MeOH', '0.5M MeOH', '0.25M MeOH', '0.1M MeOH', '0mM MeOH']
experiment.loc[3,2] = ['1M MeOH', '0.75M MeOH', '0.5M MeOH', '0.25M MeOH', '0.1M MeOH', '0mM MeOH']
experiment.loc[3,3] = ['1M MeOH', '0.75M MeOH', '0.5M MeOH', '0.25M MeOH', '0.1M MeOH', '0mM MeOH']
experiment.loc[3,4] = ['1M MeOH', '0.75M MeOH', '0.5M MeOH', '0.25M MeOH', '0.1M MeOH', '0mM MeOH']
experiment.loc[3,5] = ['1M MeOH', '0.75M MeOH', '0.5M MeOH', '0.25M MeOH', '0.1M MeOH', '0mM MeOH']
experiment.loc[3,6] = ['20mM serine', '15mM serine', '10mM serine', '5mM serine', '1mM serine', '0mM serine']
experiment.loc[3,7] = ['20mM serine', '15mM serine', '10mM serine', '5mM serine', '1mM serine', '0mM serine']
experiment.loc[3,8] = ['20mM serine', '15mM serine', '10mM serine', '5mM serine', '1mM serine', '0mM serine']

# set linestyles
linestyles = ['-','-','-','-','-','-','-','-','-','-']

# set colors
cmap = ['k', 'b', 'r','g','c','m','y','saddlebrown','orange','olive']

# calculations
DATA = g2plot.cal(DataFileName, inc_OD)

# ploting
for column in experiment:
    g2plot.plot(DATA, experiment[column], WIN, TH, Figure_Type, Yscale, 
                Xmax, Ymax, cmap, linestyles, savefig)
plt.show()