# %% [markdown]
# # This is a demo/documentation of the module.    
#   
# For the sake of performance, 
# the module is set to not do all calculations dynamically, 
# thus it is necessary to **set global parameters at the beginning**. 

# %%
# -*- coding: utf-8 -*-
from growth2fig import Experiment, Plate, Plot
# %% [markdown]
# ### Conversion factor for OD plate to OD cuvette
# The conversion factor *cvf* has to be determined experimentally. 
# According to the *Lambert-Beer law*, $A = \varepsilon * l * c$, 
# factors that change $\varepsilon$, such as wavelength & species, 
# path length $l$, such as well size & culture volume, will change the *cvf* value. 
# 
# When `cvf = 1`, the calculation takes the raw data, 
# does not calibrate to cuvette OD. 
Experiment.cvf = 0.23 

# %% [markdown]
# ### initial OD (cuvette OD)
Experiment.ini_OD = 0.02
# %% [markdown]
# ### Model for calculations. 
# The module fits Time-log(OD) to linear regression for growth parameters calculation,
# as described in [He *et al*., 2018](https://doi.org/10.1021/acssynbio.8b00093).  

# %% [markdown]
# The window size has a unit of hours. 
Experiment.WIN = 5
Experiment.TH = 0.05
# %% [markdown]
# `Plate` holds data/calculations for the whole plate.   
# 
# Give data file name (and Sheet_name, default `Sheet_name = 'Sheet1'`) to it. 
# **The data file should follow the format as the testing data.** 
# 
# Set a upper time limit for calculation and plotting if necessary.  
Experiment.plate = Plate('testingData.xlsx')
# Experiment.plate.time_upper = 100 

# %% [markdown]
# The module has three line plotting options: 
# 'all' (default), 'mean', and 'patch'. 
#
# See below their demos. 
Plot.figure_type='all'
# Plot.figure_type='mean'
# Plot.figure_type='patch'
# %% [markdown]
# The Y-scale can be set to log (default) or linear scale. 
Plot.yscale='log'
# Plot.yscale='linear'
# %% [markdown]
# The Y-axis upper limit can be set here. 
Plot.ymax=None  # None or a OD value
# %% [markdown]
# If the figures need to be saved, all formats that matplotlib accept are options. 
# The file names will be the title of the experiment.  
Plot.format=None  # 'eps','png', or None
# %% [markdown]
# ### Ways to plot
# initialize a experiment with a title,
# it will appear in the same plot. 
# 
# In legend: 
# * doubling time, time point, maximum OD 
# * SD in parentheses
# * 'NG' stands for No growth, 
# which means maximum OD does not pass the threshold (TH) set above, 
# or doubling time above 200 hours.
# 
# Note that *time point* is the start time point of the window for the calculations. 
 
exp1 = Experiment('title1')
# conditions in a list 
exp1.conditions=['condition 1', 'condition 2', 'condition 3']
# replicates (in tuples) set-ups of the experiment in a list 
# in the same of as conditions above 
exp1.repl_well_ids=[('A1','A2'), ('A3','A4'),('A5','A6')]

p1=Plot(exp1)
p1.plot()

# %% [markdown]
# ### Alternatively
exp2=Experiment('title2')
exp2.set_ups={
    'Title':'title2',
    'condition 1':('B1','B2'),
    'condition 2':('B3','B4'),
    'condition 3':('B5','B6')
}
exp2.plot()
# %% [markdown]
# ### Report growth parameters 
plate1=Plate('testingData.xlsx')
exp3 = Experiment('title 3')
exp3.plate = plate1
exp3.conditions = ['condition 1', 'condition 2', 'condition 3']
exp3.repl_well_ids = [('C1','C2'), ('C3','C4'),('C5','C6')]
exp3.result_rep
# %% [markdown]
# ### Plot in 'mean' mode 
p3=Plot(exp3)
p3.figure_type='mean'
p3.plot()

# %% [markdown]
# ### Plot in 'patch' mode
# $mean \pm SD$ 
exp4 = Experiment('title 4')
exp4.conditions=['condition 1', 'condition 2', 'condition 3']
exp4.repl_well_ids=[('D1','D2'), ('D3','D4'),('D5','D6')]
p4 = Plot(exp4)
p4.figure_type='patch'
p4.plot()
# %% [markdown]
# ### Growth parameters of the whole plate
plate1.plate_gps
# %% [markdown]
# ### Plot the whole plate 
plate1.view_plate()
# %% [markdown]
# Default `linestyles` and `colors` are as below.
# %% 
# set linestyles, preset 10 solid lines for all
Plot.linestyles = ['-','-','-','-','-','-','-','-','-','-']
# set colors, preset 10 colors
Plot.cmap = ['k', 'b', 'r','g','c','m','y','saddlebrown','orange','olive']
