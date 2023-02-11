# %%
# -*- coding: utf-8 -*-
import numpy as np 
from growth2fig import Experiment, Plate, Plot
from matplotlib import pyplot as plt 
import pylustrator 
# %% 
# global set ups
Experiment.cvf = 0.23 
Experiment.ini_OD = 0.02
Experiment.WIN = 5
Experiment.TH = 0.05

# %%
Plot.figure_type='all'
# Plot.figure_type='mean'
# Plot.figure_type='patch'
Plot.yscale='log'
# Plot.yscale='linear'
Plot.ymin=0   # 0, None, or a OD value 
Plot.ymax=None  # None or a OD value
Plot.format=None  # 'eps','png', or None
# %% 
Experiment.plate = Plate('testingData.xlsx')
# Experiment.plate.time_upper = 100 
# %%
# If you have 4 experiments/plots 
exps = [None] * 4 
 
exps[0] = Experiment('title1')
exps[1] = Experiment('title2')
exps[2] = Experiment('title3')
exps[3] = Experiment('title4')

for exp in exps: 
    exp.conditions = ['condition 1', 'condition 2', 'condition 3']

exps[0].repl_well_ids=[('A1','A2'), ('A3','A4'),('A5','A6')]
exps[1].repl_well_ids=[('B1','B2'), ('B3','B4'),('B5','B6')]
exps[2].repl_well_ids=[('C1','C2'), ('C3','C4'),('C5','C6')]
exps[3].repl_well_ids=[('D1','D2'), ('D3','D4'),('D5','D6')]

Plot.linestyles = ['-','-','-','-','-','-','-','-','--','--','--','--','--','--','--','--']
Plot.cmap = np.array([[197,1,4],[196,108,2],[175,196,2],[67,196,2],[3,195,153],[3,130,195],[4,24,195],[0,0,0],
    [197,1,4],[196,108,2],[175,196,2],[67,196,2],[3,195,153],[3,130,195],[4,24,195],[0,0,0]])/255

# df = Experiment.plate.plate_gps
# df.to_excel('plate_gps.xlsx')

pylustrator.start()
for exp in exps: 
    exp.plot()
plt.show()