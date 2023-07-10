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

# datasets_dict = cat.to_dataset_dict(
#     zarr_kwargs={'consolidated': True, 'decode_times': True})  # , preprocess=combined_preprocessing)
import dask
with dask.config.set(**{'array.slicing.split_large_chunks': False}):
    datasets_dict = cat.to_dataset_dict(zarr_kwargs={'consolidated': True, 'decode_times': True})

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
    if Path(folder + f'/tas.parq').is_file():
        continue
    Path(folder).mkdir(parents=True, exist_ok=True)
    try:
        ds = combined_preprocessing(ds)
    except:
        continue
    variables = [x for x in list(ds.keys()) if x in query['variable_id']]

    # deal with pressure levels
    if 'plev' in ds.coords:
        ds = ds.sel(plev=ds.plev.max())
    # # drop member_id if only one dimension
    # if len(list(ds.member_id)) == 1:
    #     ds = ds.drop_vars('member_id')

    # # Subset for testing purposes
    # ds = ds.isel(time=slice(None,24))
    # ds = ds.rio.write_crs('EPSG:4326')
    # ds = ds.rio.clip_box(minx=0,maxx=20,miny=30,maxy=60)
    # ds = ds.reset_coords(drop=True)  # clean up

    # Fix longitude from (0,360) to (-180,180). Has to be before clipping
    ds = ds.assign_coords({"x": (((ds.x + 180) % 360) - 180)})
    ds = ds.sortby('x')
    ds = ds.sortby('y', ascending=False)
    # # Yearly mean: average over time and longitude; then average over latitude weighting by cell area
    # global_mean = ds.groupby("time.year").mean(dim=["time", 'x'])
    # area_weight = np.cos(np.deg2rad(ds.y))  # Rectangular grid: cosine of lat is proportional to grid cell area.
    # global_mean = global_mean.weighted(area_weight).mean(dim='y')
    # global_mean = global_mean.to_dataframe()
    # for var in variables:
    #     global_mean[var].reset_index().to_parquet(folder + f'/{var}_global_mean.parq')
    # del global_mean
    # Clip to borders of countries [saves memory]
    ds = ds.rio.write_crs('EPSG:4326')
    ds = ds.rio.clip(borders.geometry.values, borders.crs, all_touched=True, drop=True)
    # Population weights are created once for every model(x temporal resolution) because all scenarios from a model have the same grid
    modelname = '.'.join([name.split('.')[i] for i in [1, 2, 4]])
    popweights_file = Path(context.projectpath() + f'/data/out/popweights/{modelname}.nc')
    if popweights_file.exists():
        ds_w = xr.open_dataset(popweights_file)
    else:
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
        ds_w.to_netcdf(popweights_file)
    # merge: add population to ds
    ds = xr.combine_by_coords([ds, ds_w])
    # assign 0 weight to (x,y,iso3) cells for which (x,y) do not fall into country iso3
    not_sea = (ds.population.notnull()) * 1
    ds['population'] = ds['population'].fillna(0)
    # Add area weight
    area_weight = np.cos(np.deg2rad(ds.y))  # Rectangular grid: cosine of lat is proportional to grid cell area.
    area_weight = area_weight * not_sea
    area_weight.name = 'area_weight'
    ds = xr.combine_by_coords([ds, area_weight])

    # keep necessary coords, vars
    ds = ds[['x', 'y', 'time'] + list(ds.keys())]

    # Aggregate
    # try:
    #     ds = agg_on_model(ds, func='mean')
    # except ValueError:
    #     # If member_id is unique and has been dropped
    #     pass
    _ds_pop = agg_on_space(ds, func='weighted_mean', weight=ds.population)
    _ds_area = agg_on_space(ds, func='weighted_mean', weight=ds.area_weight)
    _ds_area = _ds_area[variables]
    _ds_area = _ds_area.rename({x: x + '_area' for x in variables})
    ds = xr.combine_by_coords([_ds_pop, _ds_area])
    ds = agg_on_year(ds, func='mean')

    # Export
    ds = ds.drop(['population', 'area_weight'])
    ds = ds.reset_coords(drop=True)  # clean up

    # Exporting to dataframe is very slow
    with ProgressBar():
        ds = ds.compute()
    df = ds.to_dataframe()
    for var in variables:
        df[[var, var + '_area']].reset_index().to_parquet(folder + f'/{var}.parq')

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

# Convert Kelvin to Celsius
try:
    df['tas'] += -273.15
    df['tas_area'] += -273.15
except KeyError:
    pass
# Debias matching historical country temperature to UDel temp for the same period
# NB does not debias GMT and anomaly
udel = get_udel()
udel.rename(columns={'air':'udeltas'}, inplace=True)
df = pd.merge(df, udel, on=['iso3', 'year'], how='left')

bias = df[df.year.between(1980,2014)].groupby(['iso3', 'model'])[['tas', 'udeltas']].mean()
bias = (bias.tas - bias.udeltas).reset_index().rename(columns={0:'bias'})
df = pd.merge(df, bias, on=['iso3', 'model'], how='left')
df['ubtas'] = df['tas'] - df['bias']

bias_area = df[df.year.between(1980,2014)].groupby(['iso3', 'model'])[['tas_area', 'udeltas']].mean()
bias_area = (bias_area.tas_area - bias_area.udeltas).reset_index().rename(columns={0:'bias_area'})
df = pd.merge(df, bias_area, on=['iso3', 'model'], how='left')
df['ubtas_area'] = df['tas_area'] - df['bias_area']

gmt = get_gmt()
gmt = gmt.reset_index().melt(id_vars='year', var_name='scenario', value_name='gmt')

df = pd.merge(df, gmt, on=['year', 'scenario'], how='left')

for c in ['iso3', 'member_id', 'model', 'scenario']:
    df[c] = df[c].astype('category')

# Export
df.round(2).to_parquet(context.projectpath() + '/data/out/data.parq')
df.groupby(['iso3', 'year', 'model', 'scenario'])[['ubtas', 'ubtas_area']].mean().reset_index()\
    .groupby(['iso3', 'year', 'scenario'])[['ubtas', 'ubtas_area']].mean().reset_index()\
    .to_parquet(context.projectpath() + '/data/out/data_modmean.parq')

hadcrut5 = get_hadcrut5()
hadcrut5.to_parquet(context.projectpath() + '/data/out/hadcrut5.parq')

# %% Estimate coefficients with OLS
df = pd.read_parquet(context.projectpath() + '/data/out/data.parq')

# estimate
l1, l2 = [], []
for (iso3, model), g in tqdm(df.groupby(['iso3', 'model'])):
    try:
        g['gmt2'] = g.gmt**2
        res = smf.ols(formula='ubtas ~ gmt', data=g, missing='drop').fit()
        res_area = smf.ols(formula='ubtas_area ~ gmt', data=g, missing='drop').fit()
    except ValueError:
        continue
    pass
    cov_type = 'HC1'
    _ = pd.read_html(res.get_robustcov_results(cov_type=cov_type).summary().tables[1].as_html(), header=0, index_col=0)[0][['coef', 'std err']].stack()
    _ = _.to_frame().T
    _.columns = ['_'.join(x) for x in _.columns]
    _.insert(0, 'iso3', [iso3])
    _.insert(1, 'model', [model])
    _.insert(2, 'r2', [res.rsquared])
    l1.append(_)
    _ = pd.read_html(res_area.get_robustcov_results(cov_type=cov_type).summary().tables[1].as_html(), header=0, index_col=0)[0][['coef', 'std err']].stack()
    _ = _.to_frame().T
    _.columns = ['_'.join(x) for x in _.columns]
    _.insert(0, 'iso3', [iso3])
    _.insert(1, 'model', [model])
    _.insert(2, 'r2', [res.rsquared])
    l2.append(_)
coefs_ = pd.concat(l1).reset_index(drop=True)
coefs_area = pd.concat(l2).reset_index(drop=True)
coefs = pd.merge(coefs_, coefs_area, on=['iso3', 'model'], suffixes=('', '_area'))
coefs.to_parquet(context.projectpath() + '/data/out/coefficients.parq')


missing = set(df.model.unique()) - set(coefs.model.unique())
asd = df[df.model.isin(list(missing))]
asd['model'] = asd.model.astype('str')
asd['scenario'] = asd.scenario.astype('str')
asd = asd.groupby(['model', 'scenario']).count().reset_index()
pd.crosstab(asd.model, asd.scenario)

# %% Plot temperature
import matplotlib.pyplot as plt
import matplotlib as mpl
import palettable
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=palettable.wesanderson.Zissou_5.mpl_colors)

df = pd.read_parquet(context.projectpath() + '/data/out/data.parq')
df = df[df.year<=2100]

with plt.style.context('fivethirtyeight'):
    df[df.iso3 == 'ITA'].set_index('year').groupby('model').tas.plot(alpha=.25);
    # plt.legend();
    plt.show()

from statsmodels.nonparametric.smoothers_lowess import lowess
gs = df[df.iso3 == 'USA'].groupby(['iso3', 'model'])
fig, axs = plt.subplots(6,6, figsize=(20,7), sharex=True)
for i, ((iso, model), g) in enumerate(gs):
    ax = axs.flatten()[i]
    ax.set_prop_cycle('color', palettable.wesanderson.Zissou_5.mpl_colors)
    g = g[g.year.between(1900,2100)].set_index('year')
    for scen, _g in g.groupby('scenario'):
        _g.ubtas.plot(ax=ax, alpha=0.5)
    ax.plot([1980, 2014], [g.udeltas.mean(), g.udeltas.mean()], color='red')
    _g = g.groupby('year').ubtas.mean()
    _ = lowess(_g, _g.index)
    ax.plot(_[:,0], _[:,1], color='black', linestyle='dashed')
    ax.set_ylabel(model, fontsize=7)
    ax.spines[['top', 'bottom', 'right']].set_visible(False)
    # g[g.year.between(1980,2035)].set_index('year').groupby(['scenario', 'model']).tas.rolling(5).mean().plot(ax=ax)
for ax in axs.flatten()[-5:]:
    fig.delaxes(ax)
plt.suptitle('USA')
plt.tight_layout()
plt.show()

import seaborn as sns
df = df[(df.iso3=='ITA') & (df.model=='CMCC.CMCC-CM2-SR5.Amon')]
sns.lmplot(data=df, x='year', y='tas', hue='member_id')
plt.show()

colors = palettable.wesanderson.Zissou_5.mpl_colors
scenarios = df.scenario.unique()
scenarios = scenarios.categories.sort_values()
scenario_color = dict(zip(scenarios, colors))
gs = df[df.iso3 == 'USA'].groupby(['iso3', 'model'])
fig, axs = plt.subplots(6,6, figsize=(30,14), dpi=200, sharex=True, sharey=True)
for i, ((iso3, model), g1) in enumerate(gs):
    ax = axs.flatten()[i]
    g1s = g1.groupby(['scenario', 'member_id'])
    for (scenario, member_id), g2 in g1s:
        c = scenario_color[scenario]
        ax.plot(g2.year, g2.ubtas, c=c, alpha=0.35)
    _g = g1.groupby('year').ubtas.mean()
    _ = lowess(_g, _g.index)
    ax.plot(_[:, 0], _[:, 1], color='black', linestyle='dashed')
    ax.plot([1980, 2014], [g1.udeltas.mean(), g1.udeltas.mean()], color='red', zorder=2)
    ax.text(0.03, 0.97, model, fontsize=5, ha='left', va='top', transform=ax.transAxes)
# for ax in axs.flatten():
#     ax.spines[['top', 'right']].set_visible(False)
fig.subplots_adjust(hspace=0, wspace=0)
plt.tight_layout()
plt.savefig(context.projectpath() + '/img/diagnostics/USA_models.png')
plt.show()

df = df[(df.iso3 == 'USA') & (df.model=='CNRM-CERFACS.CNRM-ESM2-1.Amon')]
fig, axs = plt.subplots(1,2, sharey=True, sharex=True)
gs = df.groupby(['scenario', 'member_id'])
for (scenario, member_id), g in gs:
    c = scenario_color[scenario]
    axs[0].plot(g.year, g.tas, c=c, alpha=0.35, label=None)
    axs[1].plot(g.year, g.ubtas, c=c, alpha=0.35, label=None)
for ax in axs:
    ax.plot([1980, 2014], [df.udeltas.mean(), df.udeltas.mean()], color='red', zorder=2)
    ax.spines[['top', 'right']].set_visible(False)
axs[0].set_title('Original')
axs[1].set_title('Shifted')
plt.suptitle('CNRM-CERFACS.CNRM-ESM2-1.Amon, USA')
plt.tight_layout()
plt.savefig(context.projectpath() + '/img/diagnostics/example.png')
plt.show()
