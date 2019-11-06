import numpy as np
import pandas as pd
import time

def encode(x, enc):
    if pd.isna(x):
        return 0
    for road in enc.keys():
        if road in x:
            return enc[road]
    return 0


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ('%r  %2.2f mins' % (method.__name__, (te - ts)/60))
        return result
    return timed


class weather_FE():
    def __init__(self,df,mode='train'):
        self.data = df
        self.mode = mode

    @timeit
    def reduce_mem_usage(self,df, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024**2
        if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
        return df
    @timeit
    def fillna(self,df):
        df['air_temperature'].fillna(method='ffill',inplace=True)
        df['dew_temperature'].fillna(method='ffill',inplace=True)
        df['wind_speed'].fillna(method='ffill', inplace=True)


    def proc_pipeline(self):
        fe_data = self.reduce_mem_usage(self.data)
        return fe_data



