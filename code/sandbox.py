import intake

url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
col = intake.open_esm_datastore(url)
# loa dmultiple models
query = dict(experiment_id=['historical'],
             table_id='Amon',
             # source_id=['CMCC-CM2-HR4', 'CanESM5', 'CanESM5-CanOE', ],
             source_id=['CanESM5'],
             variable_id='pr',
             grid_label=['gn'])
cat = col.search(**query)
datasets_dict = cat.to_dataset_dict(zarr_kwargs={'consolidated': True,
                                                 'decode_times': True,
                                                 'mask_and_scale':True,
                                                 'decode_coords': True})

for name, ds in datasets_dict.items():
    pass

# W

for name, ds in datasets_dict.items():
    print(name)
    print(ds)
df = ds.isel(time=0).to_dataframe()
df.reset_index(inplace=True)
df.groupby('member_id')['pr'].describe()

for _, sl in ds.groupby('member_id'):
    print(sl)

import pandas as pd
df = pd.read_csv(r"C:\Users\Granella\Downloads\pangeo-cmip6.csv")
df = pd.read_csv('https://storage.googleapis.com/cmip6/pangeo-cmip6.csv')
for i, row in df[df.activity_id=='CMIP'].iterrows():
    row = row.drop(['dcpp_init_year'])
    cat = col.search(**row)
    print(i, len(cat))
query = df[df.activity_id=='CMIP'].iloc[0].to_dict()
{k:v for k,v in query.items() }


import xarray as xr
ds = xr.tutorial.scatter_example_dataset()
df = ds.to_dataframe().reset_index()
df.groupby('x').mean()
ds.sum(dim='x').to_dataframe().reset_index()

df = ensemble.isel(time=0).to_dataframe().reset_index()
df.p.mean()

df = pd.DataFrame(dict(lat=[0,1,2,3,4,0,1,2,3,4], lon=[0,0,0,0,0,1,1,1,1,1], v=[0,1,2,3,4,5,6,7,8,9]))
ds = df.set_index(['lon', 'lat']).to_xarray()

xr.DataArray([1,2,3], dims=['x'])
import numpy as np
arr = xr.DataArray(
    np.random.RandomState(0).randn(2, 3), [("x", ["a", "b",]), ("y", [10, 20, 30])]
)
arr
df = pd.DataFrame([[0,0,'a'],[0,0,'b'],[0,1,'a'], [0,1,np.nan]], columns=['x', 'y', 'z'])
df['v'] = np.arange(len(df))
df.set_index(['x', 'y', 'z']).to_xarray()

np.random.seed(0)
temperature = 15 + 8 * np.random.randn(2, 2, 3)
precipitation = 10 * np.random.rand(2, 2, 3)
lon = [[-99.83, -99.32], [-99.79, -99.23]]
lat = [[42.25, 42.21], [42.63, 42.59]]
time = pd.date_range("2014-09-06", periods=3)
reference_time = pd.Timestamp("2014-09-05")

ds = xr.Dataset(
    data_vars=dict(
        temperature=(["x", "y", "time"], temperature),
        precipitation=(["x", "y", "time"], precipitation),
    ),
    coords=dict(
        lon=(["x", "y"], lon),
        lat=(["x", "y"], lat),
        time=time,
        reference_time=reference_time,
    ),
    attrs=dict(description="Weather related data."),
)

population = np.arange(4).repeat(3).reshape((2,2,3))

iso3 = ['AAA', 'BBB', 'CCC']
ds_w = xr.Dataset(
    data_vars=dict(
        population=(["x", "y",'iso3'], population),
    ),
    coords=dict(
        lon=(["x", "y"], lon),
        lat=(["x", "y"], lat),
        iso3=iso3,
    ),
    attrs=dict(description="Population"),
)
ds_w.to_dataframe().reset_index().drop([1,2,3,5,6,7]).to_dict()

ds.weighted(ds_w.set_index(['x','y','iso3']).to_xarray().population)

ds_w = pd.DataFrame({'x': {0: 0, 4: 0, 8: 1, 9: 1, 10: 1, 11: 1}, 'y': {0: 0, 4: 1, 8: 0, 9: 1, 10: 1, 11: 1}, 'iso3': {0: 'AAA', 4: 'BBB', 8: 'CCC', 9: 'AAA', 10: 'BBB', 11: 'CCC'}, 'population': {0: 0, 4: 1, 8: 2, 9: 3, 10: 3, 11: 3}, 'lon': {0: -99.83, 4: -99.32, 8: -99.79, 9: -99.23, 10: -99.23, 11: -99.23}, 'lat': {0: 42.25, 4: 42.21, 8: 42.63, 9: 42.59, 10: 42.59, 11: 42.59}})
print(ds_w)  # Coordinates x,y have normally
df = pd.DataFrame({'x': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
              'y': {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 0, 7: 0, 8: 0, 9: 1, 10: 1, 11: 1},
              'iso3': {0: 'AAA', 1: 'BBB', 2: 'CCC', 3: 'AAA', 4: 'BBB', 5: 'CCC', 6: 'AAA', 7: 'BBB', 8: 'CCC', 9: 'AAA', 10: 'BBB', 11: 'CCC'},
              'population': {0: 0.0, 1: np.nan, 2: np.nan, 3: np.nan, 4: 1.0, 5: np.nan, 6: np.nan, 7: np.nan, 8: 2.0, 9: 3.0, 10: 3.0, 11: 3.0},
              'lon': {0: -99.83, 1: np.nan, 2: np.nan, 3: np.nan, 4: -99.32, 5: np.nan, 6: np.nan, 7: np.nan, 8: -99.79, 9: -99.23, 10: -99.23, 11: -99.23},
              'lat': {0: 42.25, 1: np.nan, 2: np.nan, 3: np.nan, 4: 42.21, 5: np.nan, 6: np.nan, 7: np.nan, 8: 42.63, 9: 42.59, 10: 42.59, 11: 42.59}})
print(df)

ds_w = df.set_index(['x', 'y', 'iso3']).to_xarray()
ds.weighted(ds_w.population.fillna(0)).mean()

import itertools
x = [0,1]
y = [0,1]
time = [0,1]
iso = ['A', 'B']

df = pd.DataFrame(list(itertools.product(x,y, time)), columns=['x', 'y', 'time'])
df['v'] = np.arange(len(df))
ds = df.set_index(['x', 'y', 'time']).to_xarray()

df_w = pd.DataFrame([[0,0,'A',1], [0,1,'A',1], [1,0,'A',0.5], [1,0,'B',0.5], [1,1,'B',1]], columns=['x','y','iso3','population'])
ds_w = df_w.set_index(['x', 'y', 'iso3']).to_xarray()

# fillna(0) assigns zero weight to points of coords (iso3,x,y) that do not exist (because those points of x,y coords do not fall into that iso3)
# sum(('x', 'y')): dimensions over which we want to sum. In this case, space
res = ds.weighted(ds_w.population.fillna(0))\
    .sum(('x', 'y'))\
    .to_dataframe()

print(df[df.time==0])
print(df_w)
print(res)
ds