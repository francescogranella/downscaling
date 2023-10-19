# Downscaling of global temperature anomaly

This code produces linear downscaling coefficients for temperature and (in the near future) precipitation from global level to the iso3 level. They are derived from a country-level linear regression of GLOBAL = a + b*LOCAL + e.  The following briefly explains the data sources and the bias adjustments before the regression. 

The data used to build country-level temperature comes from the subset of CMIP6 models that are _non-hot_ for [Hausfather et al. (2022)](https://www.nature.com/articles/d41586-022-01192-2). In the paper, they select 34 models whose TCR Screen (likely) in 1.4-2.2º range. At present, this downscaling excludes ACCESS-CM2 because the Pangeo file has incosistent datetime values giving problems. Coefficients for downscaling cannot be calculated for 5 additional models (DKRZ.MPI-ESM1-2-HR.Amon, DWD.MPI-ESM1-2-HR.Amon, MPI-M.MPI-ESM1-2-HR.Amon, NCAR.CESM2-FV2.Amon, NCAR.CESM2-WACCM-FV2.Amon) because they do not have historical data or do not have SSPs altogether and fail the main need. This is a known issue that might be attributable to Pangeo and may be worth addressing in the future.

The analysis is limited following the scenarios: historical, SSP126, SSP245, SSP370, SSP585. CMIP6 data is sourced from [Pangeo](https://pangeo.io/) and cleaned with the [XMIP python package](https://pypi.org/project/xmip/). The description of MIP _tables_ that can be found [here](https://clipc-services.ceda.ac.uk/dreq/index/miptable.html), whereas the list of variables and a description can be found at [this link](https://clipc-services.ceda.ac.uk/dreq/index/var.html).

The input data is gridded (at heterogeneous spatial resolution) with monthly resolution; daily resolution has been discarded as computationally too expensive. Model runs ('members') of ensembles are not averaged to maintain within-model variability.

Global mean temperature comes from two sources: [HadCRUT5](https://crudata.uea.ac.uk/cru/data/temperature/) for historical (1850-present) data and MAGICC7 for SSP-RCP projections (1995-2100). 

**Processing of CMIP data**
The first iteration of preprocessing is done with the XMIP python package, a library that has been built with standardization of CMIP files in mind. In a second stage, population weights and area weights are added. Finally, the data is aggregated on space (to country-level) weighting by population in 2000 and on time (from monthly to yearly), and exported to parquet files. 

To aggregate weighting by population, cells of CMIP6 data are split where crossed by a country border. Then, for each cell compute the population sum. It should be noted that the algorithm for computing the zonal statistic isn't exact: it uses all population pixels that *touch* a temperature pixel. It could be improved using [exactextract](https://github.com/isciences/exactextract).

**Debiasing**
Country-level temperature are debiased with respect to 1980-2014 data from the University of Delaware (originally Willmott, C. J. and K. Matsuura (2001) Terrestrial Air Temperature and Precipitation: Monthly and Annual Time Series (1950 - 1999)) [link](https://downloads.psl.noaa.gov/Datasets/udel.airt.precip/).
<img src="img/diagnostics/bias.png" alt="img/bias.png" style="zoom: 80%;" />

Some small island countries are not covered by UDel data. As such, they do not have downscaling coefficients. 
<img src="img\diagnostics\UDel_small_countries.png" style="zoom:80%;" />

HadCRUT5 and MAGICC7 time series are harmozined to form the global mean temperature variable. Specifically, HadCRUT5 is shifted to match average MAGICC7 temperature over 1995-2014. 

<img src="img\diagnostics\magicc7_hadcrut5_gap.png" style="zoom:67%;" />
<img src="img\diagnostics\magicc7_hadcrut5_nogap.png" style="zoom:67%;" />

**Regression**
After preprocessing, aggregation, and debiasing, the coefficients are estimated with $LOCAL_t = \alpha + \beta*GLOBAL_t + \epsilon_t$ separately for each country-model pair.

**Base temperatures** 
The base temperature `base_temp` is the average (debiased) local temperature over 1980-2012.