# %% import
from functions import *
import itertools
import numpy as np
import pandas as pd
import xarray as xr

# %% Create data
x = [0,1]
y = [0,1]
time = pd.date_range('2000-01-01', '2000-01-02')
member_id = ['m1', 'm2']
iso = ['A', 'B']

# Create an xarray with coordinates x,y,time, and some value v
df = pd.DataFrame(list(itertools.product(x,y, time, member_id)), columns=['x', 'y', 'time', 'member_id'])
df['v'] = np.arange(len(df))
ds = df.set_index(['x', 'y', 'time', 'member_id']).to_xarray()
# Create an array of population weights and country of location. But there is a cell that belongs to two countries.
# Hence, this xarray has 3 coordinates: x,y, and iso3.
df_w = pd.DataFrame([[0,0,'A',1], [0,1,'A',1], [1,0,'A',0.5], [1,0,'B',0.5], [1,1,'B',1]], columns=['x','y','iso3','population'])
ds_w = df_w.set_index(['x', 'y', 'iso3']).to_xarray()
# assign zero weight to points of coords (iso3,x,y) that do not exist (because those points of x,
# y coords do not fall into that iso3)
ds_w['population'] = ds_w.population.fillna(0)


# %% Test aggregation on model
df_true = ds.to_dataframe().groupby(['x', 'y', 'time'])['v'].mean().reset_index()
df_test = agg_on_model(ds, func='mean').to_dataframe().reset_index()
assert df_true.equals(df_test)
# %% Test aggregation on time
df_true = ds.to_dataframe().groupby(['x', 'y', 'member_id'])['v'].mean().reset_index()
df_test = agg_on_time(ds, func='mean').to_dataframe().reset_index().drop('year', axis=1)
assert df_true.equals(df_test)
# %% Test simple aggregation on space
df_true = ds.to_dataframe().groupby(['time', 'member_id'])['v'].mean().reset_index()
df_test = agg_on_space(ds, func='mean').to_dataframe().reset_index()
assert df_true.equals(df_test)
# %% Test weighted aggregation on space
df_true = ds.to_dataframe().reset_index().merge(ds_w.to_dataframe().reset_index()).groupby(['time', 'member_id', 'iso3']).apply(lambda x: np.average(x.v, weights=x.population)).reset_index().rename(columns={0:'v'})
df_test = agg_on_space(ds, func='weighted_mean', weight=ds_w.population).to_dataframe().reset_index()
assert df_true.equals(df_test)
# %% Test GINI aggregation on space
df_true = ds.to_dataframe().reset_index().groupby(['time', 'member_id']).apply(lambda x: gini(x.v)).reset_index().rename(columns={0:'v'})
df_test = agg_on_space(ds, func='gini').to_dataframe().reset_index()
assert df_true.equals(df_test)
# %% Test weighted GINI aggregation on space
ds = xr.combine_by_coords([ds, ds_w])
df_true = ds.to_dataframe().reset_index().merge(ds_w.to_dataframe().reset_index()).groupby(['time', 'member_id', 'iso3']).apply(lambda x: weighted_gini(x.v, x.population)).reset_index().rename(columns={0:'v'})
_ = agg_on_space(ds, func='gini', value_var='v', weight_var='population')
_.name = 'v'
df_test = _.to_dataframe().reset_index()
assert df_true.equals(df_test)
