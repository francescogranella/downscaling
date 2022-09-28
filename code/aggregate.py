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
from functions import agg_on_model, agg_on_year, agg_on_space

context.pdsettings()


# %% Function: vectorize the CMIP6 raster
def vectorize_raster(ds: xr.Dataset) -> gpd.GeoDataFrame:
    # Remove time
    ds = ds.copy().isel(time=0)
    # (half) height and width of CMIP6 raster cells
    _lon, _lat = 'x', 'y'
    h = np.diff(ds[_lat].values)[0] / 2
    w = np.diff(ds[_lon].values)[0] / 2
    # Make a vector from the CMIP6 raster
    df = ds[['y', 'x']].to_dataframe().reset_index()
    polygons = []
    for i, row in df.iterrows():
        lat = row[_lat]
        lon = row[_lon]
        polygons.append(Polygon([(lon - w, lat - h), (lon + w, lat - h), (lon + w, lat + h), (lon - w, lat + h)]))
    return gpd.GeoDataFrame(df[['x', 'y']], geometry=polygons, crs='EPSG:4326')


# %% Function: prepare the population raster for zonal statistics
def prepare_pop(path: Union[str, Path]) -> xr.DataArray:
    population = rioxr.open_rasterio(path)
    # Replace fill values
    population = population.where(population >= 0)
    # Confirm that longitude is in increasing order (W to E) and latitude is decreasing (N to S)
    # because rasterstats assumes dataset’s pixel coordinate system has its origin at the “upper left”
    population = population.sortby('x', ascending=True)
    population = population.sortby('y', ascending=False)
    band, x, y = population.indexes.values()
    assert all(x[i] <= x[i + 1] for i in range(len(x) - 1))
    assert all(y[i] >= y[i + 1] for i in range(len(y) - 1))
    # Extract the DataArray from the Dataset
    population = population.isel(band=0)
    population = population.rio.write_transform()
    return population


# %% CMIP6 files
CMIP6_simulations_paths = list(Path(context.projectpath() + '/data/in/cmip6/').rglob("*.nc"))
# %% WB borders
borders = gpd.read_file(
    context.projectpath() + '/data/in/borders/ne_10m_admin_0_countries_lakes/ne_10m_admin_0_countries_lakes.shp')
borders.loc[borders.ADMIN == 'France', 'ISO_A3'] = 'FRA'
borders.loc[borders.ADMIN == 'Norway', 'ISO_A3'] = 'NOR'
borders = borders[['ISO_A3', 'geometry']]
borders.rename(columns={'ISO_A3': 'iso3'}, inplace=True)
borders = borders[borders.iso3 != '-99']
assert borders.iso3.nunique() == len(borders)
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

# load multiple models at once
experiments = ['historical']  #, 'ssp585', 'ssp370', 'ssp245', 'ssp126']
query = dict(
    # activity_id=['CMIP'],
    experiment_id=experiments,
    table_id=['Amon'],
    source_id=screened_models,
    variable_id=['tas'],
    grid_label=['gn', 'gr', 'gr1'],
    member_id=['r1i1p1f1']
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
pbar = tqdm(datasets_dict.items())
for name, ds in pbar:
    pbar.set_description(name)
    folder = context.projectpath() + f'/data/out/cmip/{name}'
    Path(folder).mkdir(parents=True, exist_ok=True)
    try:
        ds = combined_preprocessing(ds)
    except:
        continue
    # deal with pressure levels
    if 'plev' in ds.coords:
        ds = ds.sel(plev=ds.plev.max())
    # drop member_id if only one dimension
    if len(list(ds.member_id)) == 1:
        ds = ds.drop_vars('member_id')

    # # Subset for testing purposes
    # ds = ds.sel(time=slice('2014-01-01', '2014-12-31'))
    # ds = ds.rio.write_crs('EPSG:4326')
    # ds = ds.rio.clip_box(minx=0,maxx=20,miny=30,maxy=60)
    # ds = ds.reset_coords(drop=True)  # clean up

    # Fix longitude from (0,360) to (-180,180). Has to be before clipping
    ds = ds.assign_coords({"x": (((ds.x + 180) % 360) - 180)})
    ds = ds.sortby('x')
    ds = ds.sortby('y', ascending=False)
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
    variables = [x for x in list(ds.keys()) if x in query['variable_id']]

    # Aggregate
    ds = agg_on_model(ds, func='mean')
    # # Yearly mean
    global_mean = ds.groupby("time.year").mean(dim=["time", 'x', 'y', 'iso3'])
    global_mean = global_mean.compute()
    global_mean = global_mean.to_dataframe()
    for var in variables:
        global_mean[var].reset_index().to_parquet(folder + f'/{var}_global_mean.parq')
    try:
        ds = agg_on_model(ds, func='mean')
    except ValueError:
        # If member_id is unique and has been dropped
        pass
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

# Global mean of Surface air temperature
files = list(cmip_path.glob('*/tas_global_mean.parq'))
l = []
for file in files:
    _df = pd.read_parquet(file)
    _df['model'] = '.'.join([file.parent.stem.split('.')[i] for i in [1, 2, 4]])
    _df['scenario'] = file.parent.stem.split('.')[3]
    l.append(_df)
global_mean = pd.concat(l)
global_mean.rename(columns={'tas': 'avgtas'}, inplace=True)

df = pd.merge(df, global_mean, on=['year', 'model', 'scenario'], how='outer')

# Convert Kelvin to Celsius
try:
    df['tas'] += -273.15
    df['avgtas'] += -273.15
except KeyError:
    pass
# Export
df.to_parquet(context.projectpath() + '/data/out/data.parq')
df.to_csv(context.projectpath() + '/data/out/data.csv', index=False)

# %% Estimate coefficients with OLS
df = pd.read_parquet(context.projectpath() + '/data/out/data.parq')

# remove avg GMT 1850-1899
df['avgtas'] -= df[df.year.between(1850, 1899)].groupby(['model']).avgtas.mean().values

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
