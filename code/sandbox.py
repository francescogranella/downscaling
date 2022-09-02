import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

ds = xr.tutorial.open_dataset("rasm").load()
ds.groupby('time.year').mean()
ds.mean('Tair')


month = pd.date_range('2000-01-01', '2001-12-01', freq='MS')
x = [0,1]
y = [0,1]
import itertools
df = pd.DataFrame(list(itertools.product(x,y, month)), columns=['x','y','time'])
df['v'] = np.arange(len(df))
ds = df.set_index(['x', 'y', 'time']).to_xarray()

ds.groupby('time.year').mean().to_dataframe().reset_index()
df.groupby(['x', 'y', df.time.dt.year]).mean().reset_index()

ds.groupby('time.year').mean('time').to_dataframe().reset_index()
ds.groupby('time.year').mean().to_dataframe().reset_index()


ds.groupby('time.year').mean(dim='time')