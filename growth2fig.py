# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from scipy.stats import linregress, sem
from lmfit import Model
import warnings

class Plate():
    _plate_format = 96
    def __init__(self, filename:str, cvf, ini_OD, sheet_name: str='Sheet1'):
        self.filename = filename
        self.sheet_name = sheet_name
        self._cvf = cvf
        self._ini_OD = ini_OD
            
        self.time_upper = self.time_conv().max()
        self._od = self.od_cal()
 
    @property
    def rawdata(self):
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
        return self.get_time(self.od)

    @property
    def time_upper(self):
        return self._time_upper

    @time_upper.setter
    def time_upper(self, value: float):
        if value <= 0: 
            raise ValueError('Value should be greater than 0')
        elif value > self.time_conv().max():
            warnings.warn('Time limit is greater than value in data,\
                continued.')
        else:
            self._time_upper = value
    
    @property
    def cvf(self):
        return self._cvf
   
    @property
    def ini_OD(self):
        return self._ini_OD

    @property
    def od(self):
        return self._od

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
    def plate_format(self, value: int=96):
        self._plate_format = value

    def view_plate_raw(self):
        self.plot_plate(self.rawdata, self.plate_format)
    
    def view_plate(self):
        self.plot_plate(self.od, self.plate_format)

    @property
    def wells(self):
        return self.rawdata.columns[1:]

    def get_well_gps(self, id):
        '''Growth parameters of well id '''
        return Well(id, self.od[id]).gps

    @property
    def plate_gps(self):
        '''Collect growth parameters of the whole plate'''
        self._gps = pd.DataFrame(
            index=self.wells,
            columns=[
                'Growth rate',
                'Doubling time',
                'Max OD',
                'Start time'
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
    ### return n0 * (1 + mu) ^ t # not a equivilent fuction

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
    model = 'linear'
    win = 5   # hours
    TH = 0.05

class Well:
    def __init__(self, id, df):
        self._id = id 
        self._od = df

        self._gps = self.calc_gp(self.od)
    
    # not updating !!!
    model = Experiment.model
    WIN = Experiment.win
    TH = Experiment.TH

    @staticmethod
    def calc_gp(df, WIN=WIN, model=model):
        if np.amax(df) < Well.TH: 
            gr = np.nan   # or a small number 1e-5 ?
            time_point = np.nan  # np.inf ?
        else: 
            grs = rolling_calc(df, WIN, model)
            gr = np.nanmax(grs)
            time_point = df.index.values[np.nanargmax(grs)]
        print(WIN,model)
        dt = np.log(2) / gr
        return [gr, dt, time_point]

    @property
    def id(self):
        return self._id

    @property
    def rawdata(self):
        pass 
    
    @property
    def od(self):
        return self._od 

    @property
    def max_OD(self):
        return self.od.max()
        
    @property
    def growth_rate(self):
        '''in hr-1'''
        return self._gps[0]
        
    @property
    def doubling_time(self):
        '''in hour'''
        return self._gps[1]
    
    @property
    def start_time(self):
        '''hour'''
        return self._gps[2]
    
    @property
    def gps(self):
        return [
            self.growth_rate,
            self.doubling_time,
            self.max_OD,
            self.start_time
        ]
