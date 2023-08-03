from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd
import rioxarray as rioxr
from shapely.geometry import Polygon
import xarray as xr
import matplotlib.pyplot as plt

import context

def w_avg(df, values, weights):
    d = df[values]
    w = df[weights]
    return (d * w).sum() / w.sum()


def general_agg(df: pd.DataFrame, varname: str, agg_time_first=True, time_aggfun=None, space_aggfun=None) -> pd.DataFrame:
    df = df.copy()  # Safe copy

    # If data is aggregated over time, then over space
    if agg_time_first:
        df['year'] = df.time.dt.year
        # Time agg
        if time_aggfun:
            df = df.groupby(['lat', 'lon', 'iso3', 'year'])[varname, 'population'].agg(time_aggfun).reset_index()
        # Space agg
        if space_aggfun:
            # For performance speed-up, distinguish lambda functions
            if space_aggfun.__name__ == '<lambda>':
                df = df.groupby(['iso3', 'year'])[varname, 'population'].apply(space_aggfun).reset_index()
                df.columns = ['iso3', 'year', varname]
            else:
                df = df.groupby(['iso3', 'year'])[varname, 'population'].agg(space_aggfun).reset_index()

    # If data is aggregated over space, then over time
    else:
        # Space agg
        if space_aggfun:
            # For performance speed-up, distinguish lambda functions
            if space_aggfun.__name__ == '<lambda>':
                df = df.groupby(['iso3', 'time'])[varname, 'population'].apply(space_aggfun).reset_index()
                df.columns = ['iso3', 'time', varname]
            else:
                df = df.groupby(['iso3', 'time'])[varname, 'population'].agg(space_aggfun).reset_index()
        # Time agg
        if time_aggfun:
            df['year'] = df.time.dt.year
            df = df.groupby(['iso3', 'year'])[varname].agg(time_aggfun).reset_index()

    return df[['iso3', 'year', varname]]


from xmip.preprocessing import *
def combined_preprocessing_no_unit_correction(ds):
    # fix naming
    ds = rename_cmip6(ds)
    # promote empty dims to actual coordinates
    ds = promote_empty_dims(ds)
    # demote coordinates from data_variables
    ds = correct_coordinates(ds)
    # broadcast lon/lat
    ds = broadcast_lonlat(ds)
    # shift all lons to consistent 0-360
    ds = correct_lon(ds)
    # fix the units
    # ds = correct_units(ds)
    # rename the `bounds` according to their style (bound or vertex)
    ds = parse_lon_lat_bounds(ds)
    # sort verticies in a consistent manner
    ds = sort_vertex_order(ds)
    # convert vertex into bounds and vice versa, so both are available
    ds = maybe_convert_bounds_to_vertex(ds)
    ds = maybe_convert_vertex_to_bounds(ds)
    ds = fix_metadata(ds)
    _drop_coords = ["bnds", "vertex"]
    ds = ds.drop_vars(_drop_coords, errors="ignore")
    return ds

def _simple_agg(ds, agg_var, func=None, **kwargs):
    if not func:
        return ds
    if func == 'sum':
        return ds.sum(agg_var)
    if func == 'mean':
        return ds.mean(agg_var)
    if func == 'median':
        return ds.median(agg_var)
    if func == 'std':
        return ds.std(agg_var)
    if func == 'var':
        return ds.var(agg_var)
    if func == 'max':
        return ds.max(agg_var)
    if func == 'min':
        return ds.min(agg_var)
    if func == 'quantile':
        if 'q' in kwargs:
            return ds.quantile(kwargs.get('q'))
    if func == 'gini':
        if 'weight_var' in kwargs:
            v = kwargs.get('value_var')
            w = kwargs.get('weight_var')
            return xr.apply_ufunc(
                weighted_gini,
                ds[v],
                ds[w],
                input_core_dims=[('x', 'y'), ('x', 'y')],
                vectorize=True,
                dask="parallelized"
            )
        else:
            return xr.apply_ufunc(
                gini,
                ds,
                input_core_dims=[('x', 'y')],
                vectorize=True,
                dask="parallelized"
            )
    if func == 'count_above_threshold':
        t = kwargs.get('threshold')



        return xr.apply_ufunc(
            _count_above_threshold,
            ds,
            t,
            input_core_dims=[["time"]],
            vectorize=True,
            dask='parallelized'
        )


def agg_on_year(ds, func=None, **kwargs):
    return _simple_agg(ds.groupby('time.year'), agg_var='time', func=func, **kwargs)


def agg_on_model(ds, func=None, **kwargs):
    return _simple_agg(ds, agg_var='member_id', func=func, **kwargs)


def agg_on_space(ds, func=None, weight=None, **kwargs):
    if 'weighted_' not in func:
        return _simple_agg(ds, agg_var=('x', 'y'), func=func, **kwargs)
    else:
        ds = ds.weighted(weight)
        func = func.replace('weighted_', '')
        return _simple_agg(ds, agg_var=('x', 'y'), func=func, **kwargs)


def aggregate(order='MST', *args, **kwargs):
    import itertools
    if not tuple('MST') in list(itertools.permutations(['M','S', 'T'])):
        raise ValueError('Wrong order')
    d = {'M': agg_on_model, 'S': agg_on_space, 'T': agg_on_year}
    for agg_order in list(order):
        return d[agg_order](*args, **kwargs)


def gini(x):
    sorted_x = np.sort(x)
    n = x.size
    cumx = np.cumsum(sorted_x, dtype=float)
    # The above formula, with all weights equal to 1 simplifies to:
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def weighted_gini(x, w=None):
    x = np.asarray(x).flatten()
    w = np.asarray(w).flatten()
    sorted_indices = np.argsort(x)
    sorted_x = x[sorted_indices]
    sorted_w = w[sorted_indices]
    # Force float dtype to avoid overflows
    cumw = np.cumsum(sorted_w, dtype=float)
    cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
    return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) /
            (cumxw[-1] * cumw[-1]))

def downsample(arr, factor):
    l = np.array([])
    for i, n in enumerate(arr):
        if i + 1 < len(arr):
            start, stop = n, arr[i + 1]
            l = np.concatenate([l, np.linspace(start, stop, factor + 1, endpoint=False)])
        else:
            l = np.append(l, n)
    return l


def _count_above_threshold(x, t):
    return np.sum(x >= t)


def vectorize_raster(ds: xr.Dataset) -> gpd.GeoDataFrame:
    """Vectorize raster"""
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


def prepare_pop(path: Union[str, Path]) -> xr.DataArray:
    """prepare population raster for zonal statistics"""
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


def get_borders():
    borders = gpd.read_file(
        context.projectpath() + '/data/in/borders/ne_10m_admin_0_countries_lakes/ne_10m_admin_0_countries_lakes.shp')
    borders.loc[borders.ADMIN == 'France', 'ISO_A3'] = 'FRA'
    borders.loc[borders.ADMIN == 'Norway', 'ISO_A3'] = 'NOR'
    borders = borders[['ISO_A3', 'geometry']]
    borders.rename(columns={'ISO_A3': 'iso3'}, inplace=True)
    borders = borders[borders.iso3 != '-99']
    assert borders.iso3.nunique() == len(borders)
    return borders


def get_udel():
    udel_file_path = Path(context.projectpath() + '/data/out/udel.parq')
    try:
        return pd.read_parquet(udel_file_path)
    except:
        from udel import make_udel
        return make_udel(udel_file_path, 'air')


def get_udel_gridlevel():
    udel_grid_level_file_path = Path(context.projectpath() + '/data/out/udel_grid_level.parq')
    try:
        return pd.read_parquet(udel_grid_level_file_path)
    except:
        from udel import make_udel_grid_level
        return make_udel_grid_level(udel_grid_level_file_path)


def get_hadcrut5():
    import pandas as pd
    url = 'https://crudata.uea.ac.uk/cru/data/temperature/HadCRUT5.0Analysis_gl.txt'
    df = pd.read_table(url, header=None, sep='\s+')
    df = df.iloc[(df.index % 2) == 0]
    df = df[[0,13]]
    df.columns = ['year', 'gmta']
    return df


def get_gmt(diagnostic_plots=False):
    # MAGICC7
    files = list(Path(context.projectpath() + '/data/in/magicc7').glob('SSP*'))
    l = []
    for file in files:
       _df = pd.read_csv(file)
       _df = _df[_df.variable=='Surface Temperature']
       scenario = _df.scenario.iloc[0].lower()
       idx = _df.columns.get_loc('1995')
       _df = _df[_df.columns[idx:]].T
       _df.columns = [scenario]
       l.append(_df)
    df = pd.concat(l, axis=1).reset_index().rename(columns={'index': 'year'})
    df['year'] = df.year.astype(int)


    # HadCRUT5
    hadcrut5 = get_hadcrut5()
    hadcrut5.rename(columns={'gmta':'historical'}, inplace=True)

    # Plot
    if diagnostic_plots:
        pd.merge(df, hadcrut5, on='year', how='outer').set_index('year').plot()
        plt.savefig(context.projectpath() + '/img/diagnostics/magicc7_hadcrut5_gap.png')
        plt.show()

    # Shift MAGICC7 GMT to match HadCRUT5 over 1995-2014
    # MAGICC7 GMT anomaly is the same across SSP until 2014 (observed data)
    # Mean 1995-2014
    magicc7_reference = df[df.year.between(1995,2014)].set_index('year').mean().mean()
    hadcrut5_reference = hadcrut5[hadcrut5.year.between(1995,2014)].historical.mean()
    # Gap
    gap = hadcrut5_reference - magicc7_reference
    # Close the gap
    df = df.set_index('year') + gap
    # Plot
    if diagnostic_plots:
        pd.merge(df, hadcrut5, on='year', how='outer').set_index('year').plot()
        plt.savefig(context.projectpath() + '/img/diagnostics/magicc7_hadcrut5_nogap.png')
        plt.show()

    return df