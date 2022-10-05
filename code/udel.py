# %% imports
from pathlib import Path
from typing import Union

import geopandas as gpd
import intake
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import rioxarray as rioxr
import statsmodels.formula.api as smf
import xarray as xr
from dask.diagnostics import ProgressBar
from rasterstats import zonal_stats
from shapely.geometry import Polygon
from tqdm import tqdm
from xmip.preprocessing import combined_preprocessing
import cartopy.crs as ccrs


import context
from functions import agg_on_model, agg_on_year, agg_on_space, vectorize_raster, prepare_pop, get_borders

context.pdsettings()

udel_file_path = Path(context.projectpath() + '/data/out/udel.parq')
if not udel_file_path.is_file():

    # %% Download UDel
    file_path = Path(context.projectpath() + '/data/in/udel/air.mon.mean.v501.nc')
    if not file_path.is_file():
        url = 'https://downloads.psl.noaa.gov/Datasets/udel.airt.precip/air.mon.mean.v501.nc'
        file = requests.get(url)
        with open(file_path, 'wb') as f:
            f.write(file.content)

    # %% Open and basic processing
    ds = xr.open_dataset(file_path)
    ds = ds.rename({'lat':'y', 'lon':'x', 'air':'udeltas'})
    # Fix longitude from (0,360) to (-180,180). Has to be before clipping
    ds = ds.assign_coords({"x": (((ds.x + 180) % 360) - 180)})
    ds = ds.sortby('x')
    ds = ds.sortby('y', ascending=False)
    # # %% Subset time
    # ds = ds.sel(time=slice('1980-01-01', '2014-12-31'))

    # %% Prepare population raster
    path = context.projectpath() + r"/data/in/population/gpw-v4-population-count-rev11_2000_2pt5_min_tif/gpw_v4_population_count_rev11_2000_2pt5_min.tif"
    population = prepare_pop(path)

    # %% Subset for testing purposes
    TEST = False
    if TEST:
        ds = ds.rio.write_crs('EPSG:4326')
        ds = ds.rio.clip_box(minx=-80,maxx=-60,miny=7,maxy=20)

    # %% Prepare vector
    grid = vectorize_raster(ds)
    # %% Split the grid where it crosses borders. Each cell is split into polygons, each belonging to 1 country
    borders = get_borders()
    intersections = grid.overlay(borders, how='intersection')
    # %% Zonal statistics. Calculate the population (sum/mean) in each such polygon
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
    ds['population'] = ds['population'].fillna(0)
    ds = agg_on_space(ds, func='weighted_mean', weight=ds.population)
    ds = agg_on_year(ds, func='mean')

    # Export
    ds = ds.drop('population')
    ds = ds.reset_coords(drop=True)  # clean up
    df = ds.to_dataframe()
    df.reset_index().to_parquet(udel_file_path)

