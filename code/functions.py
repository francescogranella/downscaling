import pandas as pd
import xarray as xr


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


def agg_on_time(ds, func=None, **kwargs):
    return _simple_agg(ds, agg_var='time', func=func, **kwargs)


def agg_on_model(ds, func=None, **kwargs):
    return _simple_agg(ds, agg_var='member_id', func=func, **kwargs)


def agg_on_space(ds, func=None, weight=None, **kwargs):
    if 'weighted_' not in func:
        return _simple_agg(ds, agg_var=('x', 'y'), func=func, **kwargs)
    else:
        ds = ds.weighted(weight)
        func = func.replace('weighted_', '')
        return _simple_agg(ds, agg_var=('x', 'y'), func=func, **kwargs)