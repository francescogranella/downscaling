# Notes

+ Select non-hot models following [Hausfather et al. (2022)](https://www.nature.com/articles/d41586-022-01192-2): take the models whose TCR Screen (likely) in 1.4-2.2º range. Hausfather et al. (2022) select 34 models. At present, I *exclude ACCESS-CM2* because the Pangeo file has incosistent datetime values giving problems (To do.will be fixed in the future). Coefficients for downscaling cannot be calculated for 5 additional models (DKRZ.MPI-ESM1-2-HR.Amon, DWD.MPI-ESM1-2-HR.Amon, MPI-M.MPI-ESM1-2-HR.Amon, NCAR.CESM2-FV2.Amon, NCAR.CESM2-WACCM-FV2.Amon) because they do not have historical data (and cannot be debiased), or do not have SSPs altogether and fail the main need. **This is odd, since Hausfather et al use 34 models - it could be an issue with Pangeo data Todo**. 
+ Limit analysis to the scenarios: historical, SSP126, SSP245, SSP370, SSP585.
+ Monthly data. Daily data would take too long to process
+ `tas`: Near-Surface Air Temperature [K]
+ Members (that is, runs) of ensembles are not averaged to maintain within-model variability

CMIP6 data for such models from [Pangeo](https://gallery.pangeo.io/repos/pangeo-gallery/cmip6/intake_ESM_example.html) and clean with the [XMIP python package](https://cmip6-preprocessing.readthedocs.io/en/latest/tutorial.html).

## Population weighting

Cells of CMIP6 data are split where crossed by a country border. Then, for each cell compute the population sum. ⚠️The algorithm for computing the zonal statistic isn't exact: it uses all population pixels that *touch* a temperature pixel. It can be improved using [exactextract](https://github.com/isciences/exactextract).

## Aggregation

Temperature data is aggregated over space then time with customizable functions. As of now, temperature is averaged over space weighting for population; and by year (same weight given to all months).

## Pre-processing

The goal is to estimate, separately for each country $i$ and model $m$

$T_{t} = \alpha + \beta_1 GMTanomaly_t + \beta_2 GMTanomaly^2_t + \epsilon_t \quad \forall i,m$. (1)

I estimate (1) on data pooling together different model runs (if $m$ is an ensemble) and scenarios.

❗ CMIP data is biased - does not match the observed climate. Thus  $\hat{\alpha}$  is biased. To correct this, historical temperature data is shifted, for every country-model, to match "true" University of Delaware historical temperatures over 1980-2014. The correction is then applied to projections.

1. $\Delta_{im} = T^{CMIP}_{hist,im} - T^{UDel}$
2. $\widetilde{T}_{im} = T^{CMIP}_{im} - \Delta_{im}$ $\forall$ scenarios

<img src="C:\Users\Granella\Dropbox (CMCC)\PhD\Research\impacts\img\diagnostics\example.png" alt="example" style="zoom:72%;" />

⚠️ Some small countries do not have UDel temperature. Hence, no $\widetilde{T}$. 

<img src="C:\Users\Granella\Dropbox (CMCC)\PhD\Research\impacts\img\diagnostics\UDel_small_countries.png" alt="UDel_small_countries"  />

## GMT anomaly

To match the GMT anomaly of WITCH as close as possible, I use:

1. [HadCRUT5](https://crudata.uea.ac.uk/cru/data/temperature/) for historical data.
2. MAGICC7 for SSPs

Since they are discontinuous, I'm shifting the GMT anomaly from MAGICC7 to match the HadCRUT5 (historical) over 1995-2014.  MAGICC7 GMT anomaly is the same across SSP from 1995 until 2014 (observed data).

<img src="C:\Users\Granella\Dropbox (CMCC)\PhD\Research\impacts\img\diagnostics\magicc7_hadcrut5_gap.png" alt="magicc7_hadcrut5_gap" style="zoom:67%;" /><img src="C:\Users\Granella\Dropbox (CMCC)\PhD\Research\impacts\img\diagnostics\magicc7_hadcrut5_nogap.png" alt="magicc7_hadcrut5_nogap" style="zoom:67%;" />

The HadCRUT5-MAGCC7 data is matched with de-biased CMIP data on year and scenario, and used as GMT anomaly in the regression. 



# How to

The main script is `aggregate.py`. It calls functions in `functions.py` and `context.py`.  `udel.py` downloads and cleans UDel data.

# TODO
+ [ ] Deal with inconsistencies in datetime in ACCESS-CM2.
+ [x] Replicate with non-pop weighted
+ [ ] Run with non-pop weighted
+ [x] Why do three models lack SSPs?
+ [x] Get HadCRUT5, MAGICC7. Harmonize GMT. Merge with main dataframe.
+ [x] Use HadCRUT5+MAGICC7 GMT for regressions.
+ [x] ABW (and what else?) has no UDel-derived temp. <-- Because  not covered by UDel grid
+ [x] Aggregate UDel to ISO3
+ [ ] Use population projections
+ [x] Fix regression to use UDel-adjusted temperature
+ [x] Fix regression to include different members of ensemble
+ [ ] Use exactextract
+ [ ] NUTS2 / Subnational regions
+ [x] Deal with other dimensions: plev
+ [x] Increase resolution keeping variable latitude
+ [x] Add GINI
+ [x] Check what happens when there are multiplte variables being called
+ [x] Add varname(s) to path upon saving