import pandas as pd
import matplotlib.pyplot as plt
import context

context.pdsettings()
import geopandas as gpd
import numpy as np
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
bias = pd.read_parquet(context.projectpath() + '/data/out/bias.parquet')
mbias = bias.groupby('iso3').bias.mean().reset_index()

world = world.merge(mbias, left_on='iso_a3', right_on='iso3', how='left', indicator='m')

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5), dpi=100, sharex=False, sharey=False)
cmap = plt.get_cmap('RdBu', 9)
max = np.max([np.abs(world.bias.min()), np.abs(world.bias.max())])
min = max *-1
ticks = np.linspace(min, max, 9)
world.plot(ax=ax, column='bias', legend=True, cmap=cmap, legend_kwds={"label": "$T_{CMIP} - T_{UDel}$", "orientation": "horizontal",},)
plt.title('Size of bias in CMIP6 wrt U. Delaware data')
plt.tight_layout()
plt.savefig(context.projectpath() + '/img/diagnostics/bias.png')
plt.show()