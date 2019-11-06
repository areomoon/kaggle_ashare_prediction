import numpy as np
import pandas as pd
import time

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


class dataloader():
    def __init__(self,train_df, weather_df, building_df ,mode='train'):
        self.train_df = train_df
        self.weather_df = weather_df
        self.building_df = building_df
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
    def reduce_all_df(self):
        self.train_df = self.reduce_mem_usage(self.train_df)
        self.weather_df = self.reduce_mem_usage(self.weather_df)
        self.building_df = self.reduce_mem_usage(self.building_df)

    @timeit
    def building_transform(self):
        # self.building_df.rename(columns={"square_feet": "log_square_feet"}, inplace=True)
        self.building_df['log_square_feet'] = np.float16(np.log(self.building_df['square_feet']))
        self.building_df['year_built'] = np.uint8(self.building_df['year_built'] - 1900)
        self.building_df.fillna(self.building_df['year_built'].mean(), inplace=True)  # checking
        self.building_df['floor_count'] = np.uint8(self.building_df['floor_count'])
        self.building_df.fillna(self.building_df['floor_count'].mean(), inplace=True) # checking

    @timeit
    def weather_transform(self):
        self.weather_df['air_temperature'].fillna(method='ffill',inplace=True)

    @timeit
    def merge_df(self):
        self.df = self.train_df.merge(self.building_df, on='building_id', how='left')
        self.df = self.df.merge(self.weather_df, on=['site_id', 'timestamp'], how='left')
        return self.df

    @timeit
    def time_transform(self,df):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = np.uint8(df['timestamp'].dt.hour)
        df['weekday'] = np.uint8(df['timestamp'].dt.weekday)
        df['month'] = np.uint8(df['timestamp'].dt.month)
        return df

    @timeit
    def holiday(self,df):
        holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
                    "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
                    "2017-01-02", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
                    "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
                    "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",
                    "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",
                    "2019-01-01"]
        df['is_holiday'] = (df['timestamp'].isin(holidays)).astype(int)
        return df

    @timeit
    def drop_cols(self,df):
        df = df.drop(['timestamp',"sea_level_pressure", "wind_direction", "wind_speed"], axis=1)
        return df

    def proc_pipeline(self):
        self.reduce_all_df()
        self.building_transform()
        self.weather_transform()
        df = self.merge_df()
        df = self.time_transform(df)
        df = self.drop_cols(df)

        if self.mode == 'train':
            # collect y
            label = 'meter_reading'
            y = df[label]
            df.drop([label],axis=1 ,inplace = True)
            return df, y
        else:
            df.drop("row_id", axis=1, inplace=True)
            return df


