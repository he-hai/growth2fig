# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from scipy.stats import linregress, sem
from lmfit import Model
import warnings
from functools import cached_property

class Plate():
    _plate_format = 96
    def __init__(self, filename:str, sheet_name: str='Sheet1'):
        self.filename = filename
        self.sheet_name = sheet_name
            
        self.time_upper = self.time_conv().max()
 
    @cached_property
    def rawdata(self):
        print('rawdata')
        return pd.read_excel(
            self.filename, sheet_name=self.sheet_name,
            index_col=0, engine='openpyxl',
        ).dropna()

    def time_conv(self):
        '''Convert time from seconds into hours'''
        return self.get_time(self.rawdata) / 3600

    @staticmethod
    def get_time(df: pd.DataFrame):
        return df.index.values.astype(np.float)
        
    @property
    def time(self):
        print('time')
        return self.get_time(self.od)

    @property
    def time_upper(self):
        return self._time_upper

    @time_upper.setter
    def time_upper(self, value: float):
        print('time_upper')
        if value <= 0: 
            raise ValueError('Value should be greater than 0')
        elif value > self.time_conv().max():
            warnings.warn('Time limit is greater than value in data,\
                continued.')
        else:
            self._time_upper = value
    
    @property
    def cvf(self):
        return Experiment.cvf
   
    @property
    def ini_OD(self):
        return Experiment.ini_OD

    @cached_property
    def od(self):
        print('od calc')
        return self.od_cal()
    
    def od_cal(self):
        """ OD calibration """
        if self.cvf == 1:
            od = self.rawdata
        else:
            od = self.rawdata.iloc[:,1:] - self.rawdata.iloc[:,1:5].values.min()
            od = od / self.cvf + self.ini_OD
        od.index = self.time_conv()

        return od[od.index < self.time_upper]            
    
    @staticmethod
    def plot_plate(df: pd.DataFrame, nwells: int=96):
        """ Please confirm the type of plate, 
        96-well or others."""
        if nwells == 96:
            nrows = 8
            ncols = 12
        elif nwells == 48:
            nrows = 6
            ncols = 8
        rows = ['A','B','C','D','E','F','G','H']
        cols = list(range(1,13))

        fig, axes = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True)
        fig.set_size_inches(15, 10)

        for i in range(nrows):
            for j in range(ncols):
                well = rows[i] + str(cols[j])
                if well in df.columns.values.tolist():
                    axes[i,j].plot(df.index, df.loc[:,well], linewidth=3)

        plt.show()

    @property
    def plate_format(self):
        return self._plate_format

    @plate_format.setter
    def plate_format(self, value: int):
        '''Plate format of 96 (default), 48-wells'''
        self._plate_format = value

    def view_plate_raw(self):
        self.plot_plate(self.rawdata, self.plate_format)
    
    def view_plate(self):
        print('view plate')
        self.plot_plate(self.od, self.plate_format)

    @property
    def wells(self):
        return self.rawdata.columns[1:]

    def get_well_gps(self, id):
        '''Growth parameters of well id '''
        return Well(self, id).gps

    @cached_property
    def plate_gps(self):
        '''Collect growth parameters of the whole plate.
        :Start time: the start time point of a window 
        where calculated the final growth parameters'''
        print('plate gps calc')
        self._gps = pd.DataFrame(
            index=self.wells,
            columns=[
                'Growth rate',
                'Doubling time',
                'Start time',
                'Max OD',
            ]
        )

        for id in self.wells:
            self._gps.loc[id] = self.get_well_gps(id)
        return self._gps

    @property
    def max_ODs(self):
        return self.plate_gps['Max OD']
        
    @property
    def growth_rates(self):
        # in hr-1
        return self.plate_gps['Growth rate']

    @property
    def doubling_times(self):
        # in hr
        return self.plate_gps['Doubling time']

    @property
    def start_times(self):
        # in hour
        return self.plate_gps['Start time']

def exp_gr_func(t, mu, n0):
    '''Exponential growth function
    :params t: time
    :params n0: initial OD 
    :params mu: growth rate
    '''
    return n0 * np.exp(mu * t)  # equivilent to n0*2^(t/g)
    ### return n0 * (1 + mu) ^ t # not an equivilent fuction

exp_gr_model = Model(exp_gr_func, nan_policy='omit')
exp_gr_model.set_param_hint('mu', value=1, min=1e-3)
exp_gr_model.set_param_hint('n0', value=0.02, min=1e-3)

def rolling_calc(df: pd.DataFrame, WIN, model):
    '''Rolling calculations.
    WIN: rolling window, in hours
    model options: 'linear' -> log-linear
                   'exponential' or 'log' -> exponential
    '''
    gr = np.array([])
    for tp in df[0:df.index.max() - WIN].index:
        t = df.loc[tp:tp+WIN].index.to_numpy()
        od = df.loc[tp:tp+WIN].to_numpy()

        r = gr_calc(t, od, model)
        gr = np.append(gr, r)
    return gr

def gr_calc(t, od, model):
    if model == 'linear':
        P = linregress(t, np.log(od))
        r = P.slope
    elif model == 'exponential':
        result = exp_gr_model.fit(od, t)
        r = result.params['mu'].value
    return r

class Experiment:
    '''For defining experimental set-ups and plot.
    model, WIN and TH must be defined at the beginning. 
    '''
    model = 'linear'  # for data analysis
    WIN = 5   # in hours
    TH = 0.05
    plate = None  # Plate()
    cvf = 1 
    ini_OD = 0.02

    if model == 'linear':
        model = 'linear'
    elif model == ('exponential' or 'log'):
        model = 'exponential'
    else:
        raise ValueError('Model options are:\
            linear and exponential (or log)')

    def __init__(self, title:str):
        self._title = title

    # @property
    # def plate(self):
    #     return type(self).plate
    # @plate.setter
    # def plate(self, plate: Plate):
    #     self._plate = plate

    @property
    def title(self):
        return self._title
    @title.setter
    def title(self, title:str):
        '''Experiment descriptions'''
        self._title = title
    
    @property
    def conditions(self):
        return self._conditions
    @conditions.setter
    def conditions(self, conditions:list):
        '''Treatment condition descriptions,
        in the same order of set_ups.'''
        self._conditions = conditions
        
    @property
    def repl_well_ids(self):
        return self._repl_well_ids
    @repl_well_ids.setter
    def repl_well_ids(self, repl_well_ids:list):
        '''Replicates well IDs,
        in the same order of conditions.'''
        self._repl_well_ids = repl_well_ids

    @property
    def set_ups(self):
        set_ups = {'Title':self.title}
        set_ups.update(
            dict(zip(self.conditions, self.repl_well_ids))
        )
        return set_ups
    @set_ups.setter
    def set_ups(self,set_ups:dict):
        '''set_ups = {
            'Title': 'Title x',
            'Condition 1': (well ids),
            'Condition 2': (well ids),
            ...
            }
        '''
        self._title = list(set_ups.values())[0]
        self._conditions = list(set_ups.keys())[1:]
        self._repl_well_ids = list(set_ups.values())[1:]

    @cached_property
    def result_rep(self):
        '''Result report in pd.DataFrame'''
        results = pd.DataFrame()
        for i in range(len(self.conditions)):
            result = pd.DataFrame.from_dict(
                Treatment(
                    self, self.conditions[i], self.repl_well_ids[i]
                ).result, orient='index'
            ).T
            results=pd.concat([results,result])
        return results

    def plot(self,):
        pass 

class Treatment():
    def __init__(
        self, experiment: Experiment, 
        condition: str, well_ids: tuple
    ):
        self._experiment = experiment
        self._condition = condition
        self._well_ids = well_ids 

    @property
    def experiment(self):
        return self._experiment 
    # @experiment.setter
    # def experiment(self, experiment:Experiment):
    #     self._experiment = experiment

    @property
    def plate(self):
        return self.experiment.plate

    @property
    def condition(self):
        return self._condition
    # @condition.setter
    # def condition(self, condition: str):
    #     self._condition = condition
    
    @property
    def well_ids(self):
        return self._well_ids
    # @well_ids.setter
    # def well_ids(self, well_ids: tuple):
    #     self._well_ids = well_ids

    # @property
    # def wells(self):
    #     self._wells = []
    #     for id in self.well_ids:
    #         self._well = np.append(
    #             self._wells, Well(self.plate,id)
    #         )
    #     return self._wells

    # @wells.setter
    # def wells(self, wells):
    #     self._wells = wells

    @cached_property
    def gps(self):
        return self.plate.plate_gps.loc[list(self.well_ids)]

    def calc(self, param: str):
        mean = self.gps[param].mean()
        std = self.gps[param].std()
        return (mean, std)

    @property
    def growth_rate(self):
        return self.calc('Growth rate')

    @property
    def doubling_time(self):
        return self.calc('Doubling time')

    @property
    def max_OD(self):
        return self.calc('Max OD')

    @property 
    def start_time(self):
        return self.calc('Start time')

    @property
    def result(self):
        return {
            'Experiment': self.experiment.title,
            'Condition': self.condition,
            'Wells': self.well_ids,
            'Growth rate': self.growth_rate,
            'Doubling time': self.doubling_time,
            'Max OD': self.max_OD,
            'Start time': self.start_time
        }

    @classmethod
    def plot(
        cls, color, linestyle='-', 
        figure_type='mean', yscale='log',
        ymax=None,
    ):
        '''Plot from treatments'''
        pass 

class Well:
    def __init__(self, plate: Plate, id:str):
        self._plate = plate
        self._id = id 
        
        self._gps = self.calc_gp(self.od)

    @staticmethod
    def calc_gp(df):
        if np.amax(df) < Experiment.TH: 
            gr = np.nan   # or a small number 1e-5 ?
            time_point = np.nan  # np.inf ?
        else: 
            grs = rolling_calc(df, Experiment.WIN, Experiment.model)
            gr = np.nanmax(grs)
            time_point = df.index.values[np.nanargmax(grs)]
        print(f"WIN: {Experiment.WIN}, model: {Experiment.model}, TH: {Experiment.TH}")
        dt = np.log(2) / gr
        return [gr, dt, time_point]

    @property
    def plate(self):
        return self._plate

    @property
    def id(self):
        return self._id

    @property
    def rawdata(self):
        pass 
    
    @cached_property
    def od(self):
        print(f'well od {self.id}')
        return self.plate.od[self.id]

    @property
    def max_OD(self):
        print(f'well max_OD {self.id}')
        return self.od.max()
        
    @property
    def growth_rate(self):
        '''in hr-1'''
        print(f'well gr {self.id}')
        return self._gps[0]
        
    @property
    def doubling_time(self):
        '''in hour'''
        print(f'well {self.id} doubling time')
        return self._gps[1]
    
    @property
    def start_time(self):
        '''The start time point of a window 
        where calculated the final growth parameters.
        Hour'''
        return self._gps[2]
    
    @property
    def gps(self):
        '''growth parameters'''
        return np.append(self._gps, self.max_OD)