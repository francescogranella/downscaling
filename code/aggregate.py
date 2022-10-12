# %% imports
from pathlib import Path
from typing import Union

import geopandas as gpd
import intake
import numpy as np
import pandas as pd
import rioxarray as rioxr
import statsmodels.formula.api as smf
import xarray as xr
from dask.diagnostics import ProgressBar
from rasterstats import zonal_stats
from shapely.geometry import Polygon
from tqdm import tqdm
from xmip.preprocessing import combined_preprocessing

import context
from functions import agg_on_year, agg_on_space, vectorize_raster, prepare_pop, get_borders, get_udel, get_hadcrut5, get_gmt

context.pdsettings()


# %% CMIP6 files
CMIP6_simulations_paths = list(Path(context.projectpath() + '/data/in/cmip6/').rglob("*.nc"))
# %% WB borders
borders = get_borders()
# %% Prepare population raster
path = context.projectpath() + r"/data/in/population/gpw-v4-population-count-rev11_2000_2pt5_min_tif/gpw_v4_population_count_rev11_2000_2pt5_min.tif"
population = prepare_pop(path)

# %% Download from Pangeo
url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
col = intake.open_esm_datastore(url)

# # # Get models from CSV. One model at the time
# df = pd.read_csv('https://storage.googleapis.com/cmip6/pangeo-cmip6.csv')
# for i, row in df[df.activity_id=='CMIP'].iterrows():
#     row = row.drop(['dcpp_init_year'])
#     cat = col.search(**row)

path = context.projectpath() + '/data/in/pangeo-cmip6.csv'
if Path(path).is_file():
    df = pd.read_csv(path)
else:
    df = pd.read_csv('https://storage.googleapis.com/cmip6/pangeo-cmip6.csv')
    df.to_csv(path)

source_ids = df.source_id.unique()

# Selecting non-hot models following Hausfather et al Nature 2022 https://doi.org/10.1038/d41586-022-01192-2
_models = pd.read_excel(context.projectpath() + '/data/in/cmip6/Hausfather et al Nature 2022 supporting data.xlsx',
                        skiprows=1)
_models = _models.iloc[:58]
screened_models = _models.loc[_models['TCR Screen (likely) 1.4-2.2º'] == 'Y', 'Model Name'].unique()

# Assert all screened models are in Pangeo
assert bool(set(screened_models) & set(source_ids))

screened_models = [x for x in screened_models if x!='ACCESS-CM2']
# load multiple models at once
experiments = ['historical', 'ssp585', 'ssp370', 'ssp245', 'ssp126']
query = dict(
    # activity_id=['CMIP'],
    experiment_id=experiments,
    table_id=['Amon'],
    source_id=screened_models,
    variable_id=['tas'],
    grid_label=['gn', 'gr', 'gr1'],
    # member_id=['r1i1p1f1']
)
# possible values of table_id with description  https://clipc-services.ceda.ac.uk/dreq/index/miptable.html
# possible values of variable_id with description https://clipc-services.ceda.ac.uk/dreq/index/var.html
cat = col.search(**query)

datasets_dict = cat.to_dataset_dict(
    zarr_kwargs={'consolidated': True, 'decode_times': True})  # , preprocess=combined_preprocessing)

# paths = list(Path(context.projectpath() + f'/data/in/cmip6/').rglob('*.nc'))
# for path in paths:
#     print(path.stem)
#     ds = xr.open_dataset(path)

# %% Process
# datasets = []
# for name, ds in datasets_dict.items():
#     for member_id in ds.member_id:
#         _ds = ds.sel(member_id=member_id).drop_vars('member_id')
#         newname = '.'.join([name, str(member_id.values)])
#         datasets.append((newname, _ds))

datasets_dict = dict(sorted(datasets_dict.items()))
pbar = tqdm(datasets_dict.items())
for name, ds in pbar:
    pbar.set_description(name)
    folder = context.projectpath() + f'/data/out/cmip/{name}'
    Path(folder).mkdir(parents=True, exist_ok=True)
    try:
        ds = combined_preprocessing(ds)
    except:
        continue
    variables = [x for x in list(ds.keys()) if x in query['variable_id']]
    if Path(folder + f'/tas.parq').is_file():
        continue

    # deal with pressure levels
    if 'plev' in ds.coords:
        ds = ds.sel(plev=ds.plev.max())
    # # drop member_id if only one dimension
    # if len(list(ds.member_id)) == 1:
    #     ds = ds.drop_vars('member_id')

    # # Subset for testing purposes
    # # ds = ds.sel(time=slice('2014-01-01', '2014-12-31'))
    # ds = ds.isel(time=slice(None,24))
    # ds = ds.rio.write_crs('EPSG:4326')
    # ds = ds.rio.clip_box(minx=0,maxx=20,miny=30,maxy=60)
    # ds = ds.reset_coords(drop=True)  # clean up

    # Fix longitude from (0,360) to (-180,180). Has to be before clipping
    ds = ds.assign_coords({"x": (((ds.x + 180) % 360) - 180)})
    ds = ds.sortby('x')
    ds = ds.sortby('y', ascending=False)
    # Yearly mean: average over time and longitude; then average over latitude weighting by cell area
    global_mean = ds.groupby("time.year").mean(dim=["time", 'x'])
    area_weight = np.cos(np.deg2rad(ds.y))  # Rectangular grid: cosine of lat is proportional to grid cell area.
    global_mean = global_mean.weighted(area_weight).mean(dim='y')
    global_mean = global_mean.to_dataframe()
    for var in variables:
        global_mean[var].reset_index().to_parquet(folder + f'/{var}_global_mean.parq')
    del global_mean
    # Clip to borders of countries [saves memory]
    ds = ds.rio.write_crs('EPSG:4326')
    ds = ds.rio.clip(borders.geometry.values, borders.crs, all_touched=True, drop=True)
    # Prepare vector
    grid = vectorize_raster(ds)
    # Split the grid where it crosses borders. Each cell is split into polygons, each belonging to 1 country
    intersections = grid.overlay(borders, how='intersection')
    # Calculate the population (sum/mean) in each such polygon
    stat = 'sum'
    zs = zonal_stats(vectors=intersections['geometry'], raster=population.values, affine=population.rio.transform(),
                     stats=stat,
                     nodata=np.nan, all_touched=False)
    intersections = pd.concat([intersections, pd.DataFrame(zs)], axis=1)
    intersections.rename(columns={stat: 'population'}, inplace=True)
    # to xarray
    ds_w = intersections.drop('geometry', axis=1).set_index(['y', 'x', 'iso3']).to_xarray()
    # merge: add population to ds
    ds = xr.combine_by_coords([ds, ds_w])
    # assign 0 weight to (x,y,iso3) cells for which (x,y) do not fall into country iso3
    ds['population'] = ds['population'].fillna(0)
    # keep necessary coords, vars
    ds = ds[['x', 'y', 'time'] + list(ds.keys())]

    # Aggregate
    # try:
    #     ds = agg_on_model(ds, func='mean')
    # except ValueError:
    #     # If member_id is unique and has been dropped
    #     pass
    ds = agg_on_space(ds, func='weighted_mean', weight=ds.population)
    ds = agg_on_year(ds, func='mean')

    # Export
    ds = ds.drop('population')
    ds = ds.reset_coords(drop=True)  # clean up

    # Exporting to dataframe is very slow
    with ProgressBar():
        ds = ds.compute()
    df = ds.to_dataframe()
    for var in variables:
        df[var].reset_index().to_parquet(folder + f'/{var}.parq')

# %% Combine results
cmip_path = Path(context.projectpath() + '/data/out/cmip')
# Surface air temperature
files = list(cmip_path.glob('*/tas.parq'))
l = []
for file in files:
    _df = pd.read_parquet(file)
    _df['model'] = '.'.join([file.parent.stem.split('.')[i] for i in [1, 2, 4]])
    _df['scenario'] = file.parent.stem.split('.')[3]
    l.append(_df)
df = pd.concat(l)

# # Global mean of Surface air temperature
# files = list(cmip_path.glob('*/tas_global_mean.parq'))
# l = []
# for file in files:
#     _df = pd.read_parquet(file)
#     _df['model'] = '.'.join([file.parent.stem.split('.')[i] for i in [1, 2, 4]])
#     _df['scenario'] = file.parent.stem.split('.')[3]
#     l.append(_df)
# global_mean = pd.concat(l)
# global_mean.rename(columns={'tas': 'avgtas'}, inplace=True)
#
# df = pd.merge(df, global_mean, on=['year', 'model', 'scenario', 'member_id'], how='left')

# Convert Kelvin to Celsius
try:
    df['tas'] += -273.15
    # df['avgtas'] += -273.15
except KeyError:
    pass
# Debias matching historical country temperature to UDel temp for the same period
# NB does not debias GMT and anomaly
udel = get_udel()
df = pd.merge(df, udel, on=['iso3', 'year'], how='left')
bias = df[df.year.between(1980,2014)].groupby(['iso3', 'model'])[['tas', 'udeltas']].mean()
bias = (bias.tas - bias.udeltas).reset_index().rename(columns={0:'bias'})
df = pd.merge(df, bias, on=['iso3', 'model'], how='left')
df['ubtas'] = df['tas'] - df['bias']

gmt = get_gmt()
gmt.unstack().reset_index()
gmt = gmt.reset_index().melt(id_vars='year', var_name='scenario', value_name='gmt')

df = pd.merge(df, gmt, on=['year', 'scenario'], how='left')

# Export
df.to_parquet(context.projectpath() + '/data/out/data.parq')

# %% Estimate coefficients with OLS
df = pd.read_parquet(context.projectpath() + '/data/out/data.parq')

# estimate
l = []
for (iso3, model), g in tqdm(df.groupby(['iso3', 'model'])):
    for ssp in [x for x in df.scenario.unique() if 'ssp' in x]:
        dat = g[g.scenario.isin(['historical', ssp])]
        try:
            res = smf.ols(formula='tas ~ avgtas', data=dat).fit()
        except ValueError:
            continue
        _ = pd.DataFrame({'iso3': [iso3], 'model': [model], 'scenario': ['historical_' + ssp],
                          'a': res.params['Intercept'], 'b': res.params['avgtas'],
                          'a_se': res.HC3_se['Intercept'], 'b_se': res.HC3_se['avgtas'],
                          'r2': res.rsquared
                          })
        l.append(_)
coefs = pd.concat(l).reset_index(drop=True)
coefs.to_parquet(context.projectpath() + '/data/out/coefficients.parq')
coefs.to_csv(context.projectpath() + '/data/out/coefficients.csv', index=False)
