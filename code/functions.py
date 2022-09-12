import pandas as pd
import xarray as xr


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