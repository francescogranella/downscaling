# %% Imports
from pathlib import Path
import os
import zipfile

import cdsapi

import context
context.pdsettings()
# %% Download and unzip CMIP data
query = {
    'format': 'zip',
    'temporal_resolution': 'daily',
    'experiment': 'ssp1_2_6',
    'variable': 'precipitation',
    'model': 'cmcc_esm2',
    'area': [
        70, 0, 20,
        30,
    ],
    'level': 'single_levels',
    'date': '2020-01-01/2025-12-31',
}
# Declare folders
download_folder_path = context.projectpath() + '/data/in/download.zip'
folder_name = '__'.join([query['temporal_resolution'], query['experiment'], query['variable'], query['model'], query['level'], query['date']]).replace('/', '--')
folder_path = context.projectpath() + f'/data/in/cmip6/{folder_name}'
# Download if a folder with the same name doesn't exist yet
if not os.path.isdir(folder_path):
    c = cdsapi.Client()
    c.retrieve(
        'projections-cmip6',
        query,
        download_folder_path)
    # Make folder
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    # Unzip
    with zipfile.ZipFile(context.projectpath() + '/data/in/download.zip', "r") as zip_ref:
        zip_ref.extractall(folder_path)
