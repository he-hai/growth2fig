# %%
# -*- coding: utf-8 -*-

from growth2fig import Experiment, Plate

Experiment.cvf = 0.23 
Experiment.ini_OD = 0.02
Experiment.model ='linear'
Experiment.WIN = 5
Experiment.TH = 0.05

Experiment.plate = Plate('testingData.xlsx')
# Experiment.plate.time_upper = 100 

# %%
exp1 = Experiment('title1')
exp1.conditions=['condition 1', 'condition 2', 'condition 3']
exp1.repl_well_ids=[('A1','A2'), ('A3','A4'),('A5','A6')]

exp1.result_rep
# %%
exp2=Experiment('title2')
exp2.set_ups={
    'Title':'title2',
    'condition 1':('B1','B2'),
    'condition 2':('B3','B4'),
    'condition 3':('B5','B6')
}
exp2.result_rep
# %%
p1=Plate('testingData.xlsx')
exp3 = Experiment('title 3')
exp3.plate = p1
exp3.conditions = ['condition 1', 'condition 2', 'condition 3']
exp3.repl_well_ids = [('C1','C2'), ('C3','C4'),('C5','C6')]
exp3.result_rep
# %%
