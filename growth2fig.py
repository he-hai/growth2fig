# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd 
import matplotlib
from matplotlib import pyplot as plt 
from scipy.stats import linregress
import warnings
from functools import cached_property
import re

plt.rc('axes', axisbelow=True)
# pd.set_option("display.precision", 3)
# pd.options.display.float_format = '{:.3f}'.format

class Plate():
    _plate_format = 96
    def __init__(self, filename:str, sheet_name: str='Sheet1'):
        self.filename = filename
        self.sheet_name = sheet_name
            
        self.time_upper = self.time_conv().max()
 
    @cached_property
    def rawdata(self):
        # print('rawdata')
        return pd.read_excel(
            self.filename, sheet_name=self.sheet_name,
            index_col=0, engine='openpyxl',
        ).dropna()

    def time_conv(self):
        '''Convert time from seconds into hours'''
        return self.get_time(self.rawdata) / 3600

    @staticmethod
    def get_time(df: pd.DataFrame):
        return df.index.values.astype(float)
        
    @property
    def time(self):
        # print('time')
        return self.get_time(self.od)

    @property
    def time_upper(self):
        return self._time_upper

    @time_upper.setter
    def time_upper(self, value: float):
        # print('time_upper')
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
        # print('od calc')
        return self.od_cal()
    
    def od_cal(self):
        """ OD calibration """
        if self.cvf == 1:
            od = self.rawdata
        else:
            ## blank with one well for the whole plate 
            # od = self.rawdata.iloc[:,1:] - self.rawdata.iloc[0:5,1:].values.min()
            ## blank each well individually 
            od = self.rawdata.iloc[:,1:] - self.rawdata.iloc[0:5,1:].min()
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
        # print('view plate')
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
        # print('plate gps calc')
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

def rolling_calc(df: pd.DataFrame, WIN):
    '''Rolling calculations.
    WIN: rolling window, in hours
    '''
    gr = np.array([])
    for i in range(df.size - WIN + 1):
        t = df.iloc[i:i+WIN].index.to_numpy()
        od = df.iloc[i:i+WIN].to_numpy()
        gr = np.append(gr, gr_calc(t,od))
    return gr

def gr_calc(t,od): 
    P = linregress(t, np.log(od))
    return P.slope

class Experiment:
    '''For defining experimental set-ups and plot.
    WIN and TH must be defined at the beginning. 
    '''
    WIN = 5   # in hours
    TH = 0.05
    plate = None  # Plate()
    cvf = 1 
    ini_OD = 0.02

    def __init__(self, title:str):
        self._title = title

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
        '''Replicates well IDs in a list of tuple, 
        in the same order of conditions.'''
        return self._repl_well_ids
    @repl_well_ids.setter
    def repl_well_ids(self, repl_well_ids:list):
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
    
    @property
    def treatments(self):
        _treatments=[]
        for i in range(len(self.conditions)):
            _treatments.append(
                Treatment(
                self, self.conditions[i], self.repl_well_ids[i]
            ))
        return _treatments

    @cached_property
    def result_rep(self):
        '''Result report in pd.DataFrame'''
        results = pd.DataFrame()
        for t in self.treatments:
            result = pd.DataFrame.from_dict(
                t.result, orient='index'
            ).T
            results=pd.concat([results,result])
        return results

    def plot(self):
        Plot(self).plot()

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

    @property
    def plate(self) -> Plate:
        return self.experiment.plate

    @property
    def condition(self):
        return self._condition
    
    @property
    def well_ids(self):
        return self._well_ids

    @cached_property
    def gps(self):
        _gps = pd.DataFrame(
            index=self.well_ids,
            columns=[
                'Growth rate',
                'Doubling time',
                'Start time',
                'Max OD',
            ]
        )
        for id in self.well_ids:
            _gps.loc[id] = self.plate.get_well_gps(id)
        
        _gps.loc['mean']=_gps.mean()
        _gps.loc['sd']=_gps.std()
        # print('Treatment gps calc')
        return _gps

    @property
    def growth_rate(self):
        return self.result.get('Growth rate')

    @property
    def doubling_time(self):
        return self.result.get('Doubling time')

    @property
    def max_OD(self):
        return self.result.get('Max OD')

    @property 
    def start_time(self):
        return self.result.get('Start time')

    @property
    def result(self) -> dict:
        return {
            'Experiment': self.experiment.title,
            'Condition': self.condition,
            'Wells': self.well_ids,
            'Growth rate': tuple(self.gps['Growth rate'].tail(2)),
            'Doubling time': tuple(self.gps['Doubling time'].tail(2)),
            'Max OD': tuple(self.gps['Max OD'].tail(2)),
            'Start time': tuple(self.gps['Start time'].tail(2)),
        }

    def plot(self):
        Plot(self).plot()

class Well:
    def __init__(self, plate: Plate, id:str):
        self._plate = plate
        self._id = id 

    @property
    def plate(self):
        return self._plate

    @property
    def id(self):
        return self._id
    
    @cached_property
    def od(self):
        # print(f'well od {self.id}')
        return self.plate.od[self.id]

    def get_valid_ods(self):
        _od = self.od[self.od >= Experiment.TH]
        idx1 = _od.index[0]
        idx2 = _od.index[-1]
        return self.od[idx1:idx2], idx1
    
    def get_max(self): 
        max_od = self.od.max()
        idx = self.od.idxmax()
        return max_od, idx
    
    def win_calc(self):
        WIN = np.ceil(Experiment.WIN/(
            (self.od.index[11] - self.od.index[1])/10
        ))
        if WIN < 5: 
            raise ValueError(
                'The given Experiment.WIN is too small.' + \
                f'Only {WIN} data points are in the window.'
            )
        else: 
            return int(WIN)

    def calc_gp(self):
        TH = Experiment.TH 
        WIN=self.win_calc()
        
        (max_od, idx_max) = self.get_max()
        gr = np.nan  
        time_point = np.nan
        
        if max_od >= TH: 
            (df, idx1) = self.get_valid_ods()
            
            if df[idx1:idx_max].size >= WIN: 
                grs = rolling_calc(df, WIN)
                gr = np.amax(grs)
                time_point = df.index.values[np.argmax(grs)]
            elif df[idx1:idx_max].size >= 5: 
                warnings.warn(
                    f"The given time window {Experiment.WIN} hours is probably too large for well {self.id}. " + \
                        "Use data points between TH and max OD (left side).", UserWarning
                )
                gr = gr_calc(
                    df[idx1:idx_max].index.to_numpy(),
                    df[idx1:idx_max].to_numpy()
                )
                time_point = idx1
            else: 
                return [gr, np.nan, time_point, max_od]
        dt = np.log(2) / gr
        return [gr, dt, time_point, max_od]

    @property
    def max_OD(self):
        # print(f'well max_OD {self.id}')
        return self.gps[3]
        
    @property
    def growth_rate(self):
        '''in hr-1'''
        # print(f'well gr {self.id}')
        return self.gps[0]
        
    @property
    def doubling_time(self):
        '''in hour'''
        # print(f'well {self.id} doubling time')
        return self.gps[1]
    
    @property
    def start_time(self):
        '''The start time point of a window 
        where calculated the final growth parameters.
        Hour'''
        return self.gps[2]
    
    @property
    def gps(self):
        '''growth parameters'''
        return self.calc_gp()

    def plot(self):
        Plot(self).plot()

class Plot():
    figure_type = 'all'  # 'all', 'mean' or 'patch'
    yscale = 'log'  #  'log' or 'linear'
    ymax = None  # None or OD value
    ymin = 0   # None, 0 or OD value
    format = None  # 'eps','png' or None 
    linestyles = ['-','-','-','-','-','-','-','-','-','-'] # 10 solid lines
    cmap = ['k', 'b', 'r','g','c','m',      
        'y','saddlebrown','orange','olive']   # 10 colors 

    def __init__(self, obj):
        self.obj = obj
    
    @property
    def plate(self) -> Plate:
        '''Only support ONE plate!'''
        if isinstance(self.obj, list):
        # the first obj plate
            return self.obj[0].plate 
        else:
        # Experiment, Treatment, or Well
            return self.obj.plate
        
    @property
    def x(self):  # Time 
        return self.plate.time

    @staticmethod
    def get_well_od(obj: Well):
        return obj.od.to_frame()

    @staticmethod
    def get_t_od(obj: Treatment):
        return obj.plate.od.loc[
            :,list(obj.well_ids)
        ]
    
    @classmethod
    def mean(cls, df: pd.DataFrame):
        return df.mean(axis=1)

    @classmethod
    def calc_y1(cls, df: pd.DataFrame):
        return cls.mean(df)+df.std(axis=1)
    
    @classmethod
    def calc_y2(cls, df: pd.DataFrame):
        return cls.mean(df)-df.std(axis=1)
    
    def get_od(self) -> list:
        if isinstance(self.obj, Well):
            return [self.get_well_od(self.obj)]
        elif isinstance(self.obj, Treatment):
            return [self.get_t_od(self.obj)]
        elif isinstance(self.obj, Experiment):
            od = []
            for t in self.obj.treatments:
                od.append(
                    self.get_t_od(t)
                )
            return od 

    @property 
    def y(self):  # OD
        _y = []
        if isinstance(self.obj, list):
            for obj in self.obj:
                _y += self.get_od()
        else:
            _y += self.get_od() 
        return _y

    @classmethod
    def plot_grid(cls, ax):
        for key, value in {'x':'-','y':'--'}.items():
            ax.grid(
                which='major', axis=key, linewidth=1,
                linestyle=value, color='0.75'
            )
        ax.grid(which='minor',axis='y',lw=1,ls=':',color='0.5')
    
    def formatting(self, ax):
        ax.tick_params(
            which='both', direction='in', 
            bottom=True, right=False, 
            top=False, left=True, 
            labelsize=14
        )
        ax.tick_params(
            axis='both',length=12,which='major'
        )
        ax.tick_params(
            axis='both',length=7.5,which='minor'
        )
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        ax.set_yscale(self.yscale)
        if self.yscale == 'log':
            ax.set_ylim(bottom=self.ymin,top=self.ymax)
        elif self.yscale == 'linear':
            ax.set_ylim(bottom=self.ymin,top=self.ymax)
        
        ax.set_title(self.title, fontsize=24)
        ax.set_ylabel('$OD_{600}$', fontsize=18)
        ax.set_xlabel('Time (h)', fontsize=18)
        ax.set_xlim(left=0, right=self.xmax)
   
    @property
    def xmax(self):
        return self.plate.time_upper 

    def line(self, ax, y, i):
        return ax.plot(
            self.x, y, 
            linestyle=self.linestyles[i], 
            color=self.cmap[i], 
            linewidth=3,zorder=2.02+0.02*i
        )

    def plot(self):
        fig, ax = plt.subplots(figsize=(10,8))
        lines = []
        line_num = [] 
        i = 0
        self.plot_grid(ax)
        for _y in self.y:
            line_num.append(len(lines)) 
            if self.figure_type == 'mean':
                lines += self.line(
                    ax, self.mean(_y), i
                )
            elif self.figure_type == 'all':
                lines += self.line(
                    ax, _y, i
                )
            elif self.figure_type == 'patch':
                ax.fill_between(
                    self.x, self.calc_y1(_y),
                    self.calc_y2(_y),alpha=0.2,
                    color=self.cmap[i],lw=0,
                    zorder=2.01+0.02*i
                )
                lines += self.line(
                    ax, self.mean(_y), i
                )
            i += 1

        self.formatting(ax)
        ax.legend([lines[i] for i in line_num], self.labels, fontsize=16)
        self.save()

    def save(self):
        fname = get_valid_filename(self.title)
        if self.format == 'eps':
            warnings.warn("The .eps file is fake but actually in PDF format.")
            plt.savefig(
                f'{self.figure_type}_{fname}.eps',
                dpi=300,
                format='pdf',
            )
        elif self.format:
            plt.savefig(
                f'{self.figure_type}_{fname}.{self.format}',
                dpi=300,
                format=self.format,
            )

    @property
    def labels(self):
        _labels = []
        if isinstance(self.obj, list):
            for obj in self.obj:
                _labels += self.get_label()
        else:
            _labels += self.get_label()
        return _labels

    def get_label(self) -> list:
        if isinstance(self.obj, Well):
            return [self.get_well_label(self.obj)]
        elif isinstance(self.obj, Treatment):
            return [self.get_t_label(self.obj)]
        elif isinstance(self.obj, Experiment):
            labels = []
            for t in self.obj.treatments:
                labels.append(
                    self.get_t_label(t)
                )
            return labels

    @staticmethod
    def get_well_label(obj: Well):
        gr, dt, tp, max_od = obj.gps
        if dt < 200: 
            label = f'{obj.id}: {dt:.1f}, {tp:.1f}, {max_od:.2f}'
        else:
            label = f'{obj.id}: NG, {max_od:.2f}'
        return label

    @staticmethod
    def get_t_label(obj: Treatment):
        cond = obj.condition
        dt, dt_sd = obj.doubling_time
        tp, tp_sd = obj.start_time
        max_od, m_sd = obj.max_OD
        if dt < 200:
            label = f'{cond}: {dt:.1f}({dt_sd:.1f}), {tp:.1f}' \
                + f'({tp_sd:.1f}), {max_od:.2f}({m_sd:.2f})'
        else:
            label = f'{cond}: NG, {max_od:.2f}({m_sd:.2f})'
        return label 
    
    @property
    def title(self):
        if isinstance(self.obj, Experiment):
            return str(self.obj.title)
        else:
            return None

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

def trans_cmap(cmap,alpha):
    cmap_rgba=matplotlib.colors.to_rgba_array(cmap)
    return 1-alpha*(1-cmap_rgba)