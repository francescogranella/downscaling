# %% imports
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rioxr
import intake
import xarray as xr
from mytools import pandastools
from rasterstats import zonal_stats
from shapely.geometry import Polygon
from tqdm import tqdm
from typing import Union
import context
from functions import general_agg
from xmip.preprocessing import combined_preprocessing
context.pdsettings()
pd.merge2 = pandastools.Utils.merge2

# %% functions
def w_avg(df, values, weights):
    d = df[values]
    w = df[weights]
    return (d * w).sum() / w.sum()


# %% Vectorize the CMIP6 raster
# def vectorize_raster(path: Union[str, Path]) -> gpd.GeoDataFrame:
#     ds = xr.open_dataset(path)
def vectorize_raster(ds) -> gpd.GeoDataFrame:
    # Remove time
    ds = ds.copy().isel(time=0)
    _lon, _lat = 'x', 'y'
    # (half) height and width of CMIP6 raster cells
    h = np.diff(ds[_lat].values)[0] / 2
    w = np.diff(ds[_lon].values)[0] / 2
    # Make a vector of the CMIP6 raster
    df = ds[['lat', 'lon']].to_dataframe().reset_index()
    polygons = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        lat = row[_lat]
        lon = row[_lon]
        polygons.append(Polygon([(lon-w, lat-h), (lon+w, lat-h), (lon+w, lat+h), (lon-w, lat+h)]))
    return gpd.GeoDataFrame(df[['x', 'y']], geometry=polygons, crs='EPSG:4326')


# %% Prepare the population raster for zonal statistics
def prepare_pop(path: Union[str, Path]) -> xr.DataArray:
    population = rioxr.open_rasterio(path)
    # Replace fill values
    population = population.where(population >= 0)
    # Confirm that longitude is in increasing order (W to E) and latitude is decreasing (N to S)
    # because rasterstats assumes dataset’s pixel coordinate system has its origin at the “upper left”
    population = population.sortby('x', ascending=True)
    population = population.sortby('y', ascending=False)
    band, x, y = population.indexes.values()
    assert all(x[i] <= x[i+1] for i in range(len(x) - 1))
    assert all(y[i] >= y[i+1] for i in range(len(y) - 1))
    # Extract the DataArray from the Dataset
    population = population.isel(band=0)
    population = population.rio.write_transform()
    return population

# %% CMIP6 files
CMIP6_simulations_paths = list(Path(context.projectpath() + '/data/in/cmip6/').rglob("*.nc"))
# %% WB borders
borders = gpd.read_file(context.projectpath() + '/data/in/borders/ne_10m_admin_0_countries_lakes/ne_10m_admin_0_countries_lakes.shp')
borders = borders[['SOV_A3', 'geometry']]
borders.rename(columns={'SOV_A3': 'iso3'}, inplace=True)
# %% Prepare population raster
path = context.projectpath() + r"/data/in/population/gpw-v4-population-count-rev11_2000_2pt5_min_tif/gpw_v4_population_count_rev11_2000_2pt5_min.tif"
population = prepare_pop(path)

# %% For each file, extract the grid (could vary across CMIP6 files), compute and attach population weights to it.
pass
# %% Directly from URL
url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
col = intake.open_esm_datastore(url)

# # Get models from CSV. One model at the time
# df = pd.read_csv('https://storage.googleapis.com/cmip6/pangeo-cmip6.csv')
# for i, row in df[df.activity_id=='CMIP'].iterrows():
#     row = row.drop(['dcpp_init_year'])
#     cat = col.search(**row)

# load multiple models at once
query = dict(experiment_id=['historical'],
             table_id='Amon',
             source_id=['CMCC-CM2-HR4', 'CanESM5', 'CanESM5-CanOE', ],
             variable_id='pr',
             grid_label=['gn'])
cat = col.search(**query)
datasets_dict = cat.to_dataset_dict(zarr_kwargs={'consolidated': True, 'decode_times': True})
for name, ds in datasets_dict.items():
    pass


# %% From disk
# for path in CMIP6_simulations_paths:
#     ds = xr.open_dataset(path)

    # Aggregate CMIP6 data using a function of choice over time and space
    ds = combined_preprocessing(ds)
    # Subset for testing purposes
    ds = ds.rio.write_crs('EPSG:4326')
    ds = ds.rio.clip_box(minx=10,maxx=20,miny=30,maxy=40)
    # Prepare vector
    grid = vectorize_raster(ds)
    # Zonal statistics: mean population density per CMIP6 cell
    zs = zonal_stats(vectors=grid['geometry'], raster=population.values, affine=population.rio.transform(), stats='sum', nodata=np.nan, all_touched=True)
    weight_vector = pd.concat([grid, pd.DataFrame(zs)], axis=1)
    weight_vector.rename(columns={'sum': 'population'}, inplace=True)
    #  Spatial join: weight polygons in national borders
    weight_vector = gpd.sjoin(weight_vector, borders, predicate='intersects', how='inner')
    ds_w = weight_vector.drop(['geometry', 'index_right'], axis=1).set_index(['x', 'y', 'iso3']).to_xarray()
    # assign 0 weight to (x,y,iso3) cells for which (x,y) do not fall into country iso3
    ds_w['population'] = ds_w['population'].fillna(0)

    # merge: add population  to ds
    ds = xr.combine_by_coords([ds, ds_w])

    from functions import *
    ds = agg_on_model(ds, func='mean')
    ds = agg_on_time(ds, func='mean')
    agg_on_space(ds, func='weighted_mean', weight=ds.population)


    # Save metadata
    with open(context.projectpath() + f'/data/out/{path.stem}.txt', 'w') as f:
        print(ds.attrs, file=f)

