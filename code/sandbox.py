# %% imports

import geopandas as gpd
import matplotlib.pyplot as plt
from mytools import pandastools

import context
from functions import *

context.pdsettings()

path = context.projectpath() + '/data/in/borders/WB_GAD_Med/WB_GAD_ADM1.shp'
gdf = gpd.read_file(path)
gdf.boundary.plot(figsize=(20, 20));
plt.show()

ds = xr.tutorial.load_dataset("air_temperature").isel(time=0)
ds.air.plot()
plt.show()

# Increase resolution
factor = 2
lat = ds.lat.values
lat_step = np.mean(np.diff(lat)) / factor
new_lat = np.arange(lat.max(), lat.min(), lat_step)
lon = ds.lon.values
lon_step = np.mean(np.diff(lon)) / factor
new_lon = np.arange(lon.min(), lon.max(), lon_step)
# with interp, provide new lat and lon
ds.interp(lat=new_lat, lon=new_lon).air.plot()
plt.show()


#