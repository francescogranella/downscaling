# %% imports
from pathlib import Path
from typing import Union

import dask
import geopandas as gpd
import intake
import rioxarray as rioxr
from rasterstats import zonal_stats
from shapely.geometry import Polygon
from tqdm import tqdm

import context
from functions import *

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
    df = ds[['lat', 'lon']].to_dataframe().reset_index()
    polygons = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
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
borders = borders[['ISO_A3', 'geometry']]
borders.rename(columns={'ISO_A3': 'iso3'}, inplace=True)
borders = borders[borders.iso3!='-99']
assert borders.iso3.nunique() == len(borders)
# %% Prepare population raster
path = context.projectpath() + r"/data/in/population/gpw-v4-population-count-rev11_2000_2pt5_min_tif/gpw_v4_population_count_rev11_2000_2pt5_min.tif"
population = prepare_pop(path)

# %% For each file, extract the grid (could vary across CMIP6 files), compute and attach population weights to it.
pass
# %% Download from Pangeo
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
             source_id=['CMCC-CM2-HR4'],
             variable_id=['ta'],
             grid_label=['gn'])
cat = col.search(**query)
with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    datasets_dict = cat.to_dataset_dict(zarr_kwargs={'consolidated': True, 'decode_times': True})

for name, ds in datasets_dict.items():
    # Aggregate CMIP6 data using a function of choice over time and space

    ds = combined_preprocessing(ds)
    # Fix longitude from (0,360) to (-180,180). Has to be before clipping
    ds['x'] = ds.x.where(ds.x <= 180, ds.x - 360)
    ds = ds.sortby('x')
    ds = ds.sortby('y', ascending=False)
    # Clip to borders of countries [saves memory]
    ds = ds.rio.write_crs('EPSG:4326')
    ds = ds.rio.clip(borders.geometry.values, borders.crs, all_touched=True, drop=True)
    ds.isel(time=0, plev=0, member_id=0).ta.plot(robust=True);
    plt.show()
    # # Subset for testing purposes
    # ds = ds.rio.clip_box(minx=0,maxx=60,miny=0,maxy=60)
    # Prepare vector
    grid = vectorize_raster(ds)
    factor = 2
    y = ds.y.values
    new_y = downsample(y, factor)
    x = ds.x.values
    new_x = downsample(x, factor)
    x_step = np.mean(np.diff(x)) / factor
    if x_step >= 1:
        ds = ds.interp(y=new_y, x=new_x)
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
    if 'plev' in ds.coords:
        ds = ds.isel(plev=0)
    # Aggregate
    ds = agg_on_model(ds, func='mean')
    ds = agg_on_space(ds, func='weighted_mean', weight=ds.population)
    ds = agg_on_year(ds, func='mean')

    # Export
    variables = '-'.join([x for x in list(ds.keys()) if x != 'population'])
    name = '.'.join([name, variables])
    ds.to_dataframe().reset_index().to_parquet(context.projectpath() + f'/data/out/{name}.parq')

    # Visual sanity check
    do = False
    if do:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        world.loc[world.name == 'France', 'iso_a3'] = 'FRA'
        world.loc[world.name == 'Norway', 'iso_a3'] = 'NOR'
        df = ds.isel(year=-1).to_dataframe().reset_index()
        gdf = world.merge(df, left_on='iso_a3', right_on='iso3')
        v = query['variable_id'][0]
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(constrained_layout=True, dpi=200, figsize=(20, 20))
        gdf.plot(column=v, legend=True)
        ax.set_title(v)
        plt.show()
