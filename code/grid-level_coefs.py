# %% imports
import glob
from pathlib import Path

import intake
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf
import xarray as xr
from tqdm import tqdm
from xmip.preprocessing import combined_preprocessing

import context
from functions import agg_on_year, prepare_pop, get_borders, get_gmt
from functions import get_udel_gridlevel

context.pdsettings()

import warnings

warnings.filterwarnings("ignore")

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

screened_models = [x for x in screened_models if x != 'ACCESS-CM2']
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

# datasets_dict = cat.to_dataset_dict(
#     zarr_kwargs={'consolidated': True, 'decode_times': True})  # , preprocess=combined_preprocessing)
import dask

with dask.config.set(**{'array.slicing.split_large_chunks': False}):
    datasets_dict = cat.to_dataset_dict(zarr_kwargs={'consolidated': True, 'decode_times': True})

# %% Process
start_bias_window = 1980
# datasets_dict = dict(sorted(datasets_dict.items()))
datasets_dict = {x[1]: x[2] for x in sorted([(k.split('.')[1:], k, v) for k, v in datasets_dict.items()])}
pbar = tqdm(datasets_dict.items())
for name, ds in pbar:
    pbar.set_description(name)
    # if not 'BCC.BCC-CSM2-MR' in name:
    #     continue
    # else:
    #     print(name)
    folder = context.projectpath() + f'/data/out/grid-level/{name}'
    if Path(folder + f'/tas.parquet').is_file():
        continue
    Path(folder).mkdir(parents=True, exist_ok=True)
    try:
        ds = combined_preprocessing(ds)
    except:
        continue
    variables = [x for x in list(ds.keys()) if x in query['variable_id'] and x not in ['height', 'areacella']]
    ds = ds[variables]
    # deal with pressure levels
    if 'plev' in ds.coords:
        ds = ds.sel(plev=ds.plev.max())

    ds = ds.where(ds.time.dt.year >= 1980, drop=True)
    # Fix longitude from (0,360) to (-180,180). Has to be before clipping
    ds = ds.assign_coords({"x": (((ds.x + 180) % 360) - 180)})
    ds = ds.sortby('x')
    ds = ds.sortby('y', ascending=False)
    ds = ds.drop(['lat', 'lon'])
    # Coarsen the grid
    grid = xr.DataArray(dims=('y', 'x'), coords={'y': np.arange(-90, 90, 0.5), 'x': np.arange(-180, 180, 0.5)})
    ds = ds[['x', 'y', 'time'] + list(ds.keys())]
    ds = agg_on_year(ds, func='mean').interp_like(grid, method='nearest')
    # Clip to borders of countries [saves memory]
    ds = ds.rio.write_crs('EPSG:4326')
    ds = ds.rio.clip(borders.geometry.values, borders.crs, all_touched=True, drop=True)

    # Mean over member_id - not used
    ds = ds.mean(dim='member_id')

    # Convert Kelvin to Celsius
    try:
        ds['tas'] += -273.15
    except KeyError:
        pass

    # Exporting to dataframe is very slow
    # with ProgressBar():
    ds = ds.compute()
    df = ds.to_dataframe().dropna(subset=['tas'])
    df = df[['tas']].astype(float).round(2)

    pl.from_pandas(df.reset_index()).write_parquet(folder + f'/tas.parquet')
    del ds, df

# %% Combine results
cmip_path = Path(context.projectpath() + '/data/out/grid-level')
models = set(['.'.join([x.name.split('.')[i] for i in [1, 2]]) for x in list(cmip_path.glob('*'))])
udel = get_udel_gridlevel().dropna(subset=['udeltas'])
udel_mean = udel[udel.year.between(1980, 2014)].groupby(['x', 'y']).udeltas.mean().reset_index().round(2)
del udel
gmt = get_gmt()
gmt = gmt.reset_index().melt(id_vars='year', var_name='scenario', value_name='gmt').round(2)

# Break up by model to save memory
pbar = tqdm(models)
for model in pbar:
    pbar.set_description(model)
    model_data_path = Path(context.projectpath() + f'/data/out/grid-level/{model}_data.parquet')
    if model_data_path.is_file():
        continue
    files = list(cmip_path.glob(f'*{model}.*/tas.parq*'))  # assumes only Amon
    if len(files) != 5:
        continue
    l = []
    for file in tqdm(files):
        _df = pd.read_parquet(file)
        _df = _df[_df.year <= 2100]
        _df.insert(0, 'scenario', file.parent.stem.split('.')[3])
        l.append(_df)
        del _df
    df = pd.concat(l, ignore_index=True, sort=False)
    del l
    df['scenario'] = df.scenario.astype('category')
    df.set_index(['x', 'y'], inplace=True)
    # Debias matching historical country temperature to UDel temp for the same period
    # NB does not debias GMT and anomaly
    bias = pd.merge(df[df.year.between(1980, 2014)],
                    udel_mean.set_index(['x', 'y']),
                    left_index=True, right_index=True, how='inner').groupby(['x', 'y'])[['tas', 'udeltas']].mean()
    bias = (bias.tas - bias.udeltas).round(2).reset_index().rename(columns={0: 'bias'}).set_index(['x', 'y'])
    df = pd.merge(df, bias, left_index=True, right_index=True, how='inner')
    df['ubtas'] = df['tas'] - df['bias']
    df = df[['year', 'scenario', 'ubtas']].reset_index()

    # TODO do not merge with GMT. Add just before OLS
    df = pd.merge(df, gmt, on=['year', 'scenario'], how='inner')

    df.round(2).to_parquet(model_data_path)
    del df

# %% Coefficients
path = Path(r"C:\Users\Granella\Dropbox (CMCC)\PhD\Research\impacts\data\out\grid-level")
model_data_paths = list(path.glob('*_data.parquet'))


def ols(i, data):
    x, y = i
    res = smf.ols(formula='ubtas ~ gmt', data=data, missing='drop').fit()
    table = res.summary2().tables[1]
    table['r2'] = res.rsquared
    table['x'] = x
    table['y'] = y
    return table


for model_data_path in model_data_paths:
    model = model_data_path.stem[:-5]
    print(model)
    out_path = Path(context.projectpath() + f'/data/out/grid-level/{model}_coefficients.parquet')
    if not out_path.is_file():
        df = pd.read_parquet(model_data_path)
        df = df.drop(columns=['year', 'scenario'])
        gs = df.groupby(['x', 'y'])

        # parallelize the OLS regression on multiple datasets
        results = Parallel(n_jobs=4)(delayed(ols)(i, dataset) for (i, dataset) in tqdm(gs))
        coefs = pd.concat(results)
        coefs.to_parquet(out_path)
    else:
        continue

# %% All coefficients in one dataset
path = Path(context.projectpath() + f'/data/out/grid-level')
coefficient_paths = list(path.glob('*_coefficients.parquet'))

l = []
for coefficient_path in coefficient_paths:
    coefs = pd.read_parquet(coefficient_path).reset_index()[['x', 'y', 'index', 'Coef.']] \
        .pivot(index=['x', 'y'], columns=['index'], values='Coef.') \
        .reset_index() \
        .rename(columns={'x': 'lon', 'y': 'lat', 'Intercept': 'intercept'})
    model = coefficient_path.stem[:-13]
    coefs['model'] = model
    l.append(coefs)

df = pd.concat(l, sort=False, ignore_index=True)

df.round(2).to_parquet(context.projectpath() + f'/data/out/grid-level/coefficients.parquet')

# %% Model mean temperature
path = Path(r"C:\Users\Granella\Dropbox (CMCC)\PhD\Research\impacts\data\out\grid-level")
model_data_paths = list(path.glob('*_data.parquet'))

_df = pl.read_parquet(model_data_paths[0], columns=['scenario'])

l = []
for scenario in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
    l2 = []
    for model_data_path in tqdm(model_data_paths):
        _df = pl.read_parquet(model_data_path, columns=['year', 'x', 'y', 'ubtas', 'scenario'])\
            .filter(pl.col('scenario') == scenario)\
            .select(['x', 'y', 'year', 'ubtas'])\
            .with_column(pl.col('year').cast(pl.Int64, strict=False))

        l2.append(_df)

    _df = pl.concat(l2)\
        .groupby(['x', 'y', 'year'])\
        .agg(pl.mean('ubtas')).with_column(pl.lit(scenario).alias('scenario'))
    l.append(_df)

df = pl.concat(l)
df.write_parquet(context.projectpath() + f'/data/out/grid-level/data_modmean.parquet')

# %% Model mean coefficients
gmt = get_gmt()
gmt = gmt.reset_index().melt(id_vars='year', var_name='scenario', value_name='gmt').round(2)

gs = pd.read_parquet(context.projectpath() + f'/data/out/grid-level/data_modmean.parquet')\
    .merge(gmt, on=['year', 'scenario'], how='inner')\
    .drop(columns=['year', 'scenario'])\
    .groupby(['x', 'y'])

# parallelize the OLS regression on multiple datasets
results = Parallel(n_jobs=4)(delayed(ols)(i, dataset) for (i, dataset) in tqdm(gs))
coefs = pd.concat(results)
coefs = coefs.reset_index()[['x', 'y', 'index', 'Coef.']]\
    .pivot(index=['x', 'y'], columns=['index'], values='Coef.') \
    .reset_index() \
    .rename(columns={'x': 'lon', 'y': 'lat', 'Intercept': 'intercept'})
coefs.round(2).to_parquet(context.projectpath() + f'/data/out/grid-level/coefficients_modmean.parquet')

# %% Plot
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

path = Path(context.projectpath() + f'/data/out/grid-level')
coefficient_paths = list(path.glob('*_coefficients.parq'))
coefficient_paths.append(Path(context.projectpath() + f'/data/out/grid-level/coefficients_modmean.parquet'))

coefficient_path = Path(context.projectpath() + f'/data/out/grid-level/coefficients_modmean.parquet')
coefs_ds = pd.read_parquet(coefficient_path).set_index(['lat', 'lon']).to_xarray()
coefs_ds = coefs_ds.rename({'gmt':'slope'})

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5),
                        sharex=False, sharey=False, subplot_kw={'projection': ccrs.Robinson()},
                        constrained_layout=False, dpi=250
                        )
for ax, var in list(zip(axs, ['slope', 'intercept'])):
    if var == 'slope':
        vmin, vcenter, vmax = -0, 1, 2
    else:
        vmin, vcenter, vmax = -40, 0, 40
    nlevels = 9
    ticks = np.round(np.linspace(vmin, vmax, nlevels), 1)
    ax.set_global()
    cbar_kwargs = {'orientation': 'horizontal', 'shrink': 1., 'aspect': 80, 'label': '', 'ticks': ticks}
    coefs_ds[var].plot.contourf(ax=ax, levels=nlevels, transform=ccrs.PlateCarree(), robust=True,
                                cbar_kwargs=cbar_kwargs, cmap='coolwarm',
                                vmin=vmin, vcenter=vcenter, vmax=vmax)
    ax.coastlines()
    ax.set_title(var, fontsize=20)
    # plt.suptitle('Downscaling slope')
plt.tight_layout()
plt.show()


# One big figure
ncols = int(np.floor(len(coefficient_paths) ** 0.5))
nrows = int(np.ceil(len(coefficient_paths) / ncols))
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 5),
                        sharex=False, sharey=False, subplot_kw={'projection': ccrs.Robinson()},
                        constrained_layout=True, dpi=250
                        )
for ax, coefficient_path in zip([axs], coefficient_paths):
    if coefficient_path.stem != 'coefficients_modmean':
        coefs = pd.read_parquet(coefficient_path).reset_index()
        coefs_ds = coefs[coefs['index'] == 'gmt'].set_index(['y', 'x']).to_xarray()
        model = coefficient_path.stem[:-13]
        var = 'Coef.'
    else:
        coefs_ds = pd.read_parquet(coefficient_path, columns=['lon', 'lat', 'gmt']).set_index(['lat', 'lon']).to_xarray()
        model = 'Model mean'
        var = 'gmt'
    # coefs_ds = coefs_ds.sel(lon=slice(-10,20),lat=slice(36,66))
    vmin, vcenter, vmax = -1, 1, 3
    nlevels = 9
    ticks = np.round(np.linspace(vmin, vmax, nlevels), 1)

    # ax.set_global()
    cbar_kwargs = {'orientation': 'horizontal', 'shrink': 0.6, 'aspect': 40, 'label': '', 'ticks': ticks}
    coefs_ds[var].plot.contourf(ax=ax, levels=nlevels, transform=ccrs.PlateCarree(), robust=True,
                                    cbar_kwargs=cbar_kwargs, cmap='coolwarm',
                                    vmin=vmin, vcenter=vcenter, vmax=vmax)
    ax.coastlines()
    ax.set_title(model, fontsize=5)
plt.suptitle('Downscaling slope')
plt.show()

# Multiple figures
for coefficient_path in coefficient_paths:
    coefs = pd.read_parquet(coefficient_path).reset_index()
    model = coefficient_path.stem[:-13]
    coefs = coefs[coefs['index'] == 'gmt']

    coefs_ds = coefs.set_index(['y', 'x']).to_xarray()

    vmin, vcenter, vmax = -1, 1, 3
    nlevels = 9
    ticks = np.round(np.linspace(vmin, vmax, nlevels), 1)

    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    cbar_kwargs = {'orientation': 'horizontal', 'shrink': 0.6, 'aspect': 40, 'label': 'Slope', 'ticks': ticks}
    coefs_ds['Coef.'].plot.contourf(ax=ax, levels=nlevels, transform=ccrs.PlateCarree(), robust=True,
                                    cbar_kwargs=cbar_kwargs, cmap='coolwarm',
                                    vmin=vmin, vcenter=vcenter, vmax=vmax)
    ax.coastlines()
    plt.title(model)
    plt.show()

# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Generate some example data
data = xr.DataArray(np.random.rand(10, 10), dims=['x', 'y'])

# Define the colormap and normalization
cmap = plt.cm.jet
bounds = np.linspace(0, 1, 11)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Plot the data using xarray's built-in plotting function
data.plot(cmap=cmap, norm=norm)

# Add a colorbar with custom tick locations and labels
plt.contourf(data)
cbar = plt.colorbar(ticks=bounds)
cbar.ax.set_yticklabels(['{:.1f}'.format(b) for b in bounds])
plt.show()
