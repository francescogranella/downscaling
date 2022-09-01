# Functions and scripts to compute country-level data of spatial variables (e.g., climate)


#get gridded climate variables

#require(rworldmap)
#require(Rnightlights)
# Extracting CRU Climate data: CRU TS v4.01  (0.5° grid or about 50kmx50km cells)
# Complete guide available at: http://www.benjaminbell.co.uk

#CMIP5 RCPs 1861-2100 Ensemble Mean from Shouro asgupta


#Gridded population as of 2000 from SEDAC @ NASA
# http://sedac.ciesin.columbia.edu/data/set/gpw-v3-population-density/data-download



################ SETUP ###################
#CHoose "all" or a specific iso3 for country specific processing (if one iso3, will also saved as csv)
iso3_save = "all"
#iso3_save = "ZAF"
#iso3_save = c("FRA", "DEU", "ITA", "ESP")


dataset_name = "rcp_dasgupta" #|"cru_2000s", "cru_1901_2016", "rcp_dasgupta"

pop2000weighted=F

##########################################







library(raster)
library(ncdf4)
library(tidyverse)
#require(data.table)
library(countrycode)
require(GADMTools)
require(rgdal)

#Function to aggregate spatial data to country level data as (weighted) average
raster2iso3 <- function(rasterdata, iso3="all", debug=F){
  var_category <- deparse(substitute(rasterdata))
  #rasterdata <- rotate(rasterdata) #since the shapefiles are -180 - +180, not 0 - 360 degrees!
  if(pop2000weighted){
  #first multiply all values by grid population
  test <- rasterdata[[names(rasterdata)]] * population_grid[[1]]
  names(test) <- names(rasterdata)
  rasterdata <- test
  } 
  if(iso3[1]=="all") {
  iso3 <- countrycode::codelist$iso3c; iso3 <- iso3[!is.na(iso3)] #better, based on country code
  iso3 <- setdiff(iso3, c("ATA"))
  }
  for(.iso3 in iso3){
    print(str_glue("Processing {.iso3} for variable {var_category} in RCP {rcp}"))
    #gadm.loadCountries("BRA", level = 0, basefile=gadm36, simplify=NULL)
    countrygeom <- gadm36_0_shp[gadm36_0_shp$GID_0==.iso3,]
    #countrygeom <- readRDS(paste0("old_gadm", "//GADM_2.8_",.iso3,"_adm0.rds"), refhook = NULL)
    if(debug){plot(rasterdata[[1]]); plot(countrygeom, bg="transparent", add=TRUE)}
  if(pop2000weighted){
     .countrydata <- raster::extract(rasterdata, countrygeom, fun = sum, na.rm = TRUE, sp = TRUE)
     .countrypop <- raster::extract(population_grid, countrygeom, fun = sum, na.rm = TRUE, sp = TRUE)
     if(length(names(.countrydata))>1){
       .varnames <- setdiff(names(.countrydata), c("NAME_0", "GID_0"))
       .countrydata <- as.data.frame(.countrydata)[.varnames]
       .countrydata <- .countrydata / .countrypop[[3]] #now divide by total population to get average
       .countrydata$iso3 <- .iso3
       if(!exists("allcountrydata")){allcountrydata <- .countrydata}else{allcountrydata <-rbind(allcountrydata, .countrydata)}
     }
   }else{
     .countrydata <- raster::extract(rasterdata, countrygeom, fun = mean, na.rm = TRUE, sp = TRUE)
     if(length(names(.countrydata))>1){
       .varnames <- setdiff(names(.countrydata), c("NAME_0", "GID_0"))
       .countrydata <- as.data.frame(.countrydata)[.varnames]
       .countrydata$iso3 <- .iso3
       if(!exists("allcountrydata")){allcountrydata <- .countrydata}else{allcountrydata <-rbind(allcountrydata, .countrydata)}
     }
   }
  }
  return(allcountrydata)
}

#function to convert monthly series in annual min/max/averages (in celsius and monthly mm precipitation)
month2year <- function(data, varname, startyear, unit="celsius"){
  #require(reshape); data2 <- melt(data, id="iso3")
  data2 <- data %>% gather(., variable, value, -iso3) %>% as.data.frame()
  if(unit=="kelvin") data2$value <- data2$value - 273.15
  if(unit=="inch") data2$value <- data2$value * 25.4
  if(unit=="kg/m2/s") data2$value <- data2$value * (60*60*24*30)
  #data2sum <- data2 %>% mutate(year=as.integer(substr(variable, 2,5))) %>% group_by(iso3, year) %>% summarize(mean=mean(value), min=min(value), max=max(value)) #works for UEA-GRU
  data2sum <- data2 %>% arrange(iso3, variable) %>% group_by(iso3) %>% mutate(year=startyear+((row_number()-1) %/% 12), month=((row_number()-1) %% 12)+1) %>% ungroup() %>% group_by(iso3, year) %>% summarize(mean=mean(value), min=min(value), max=max(value))
  data2sum <- as.data.frame(data2sum)
  colnames(data2sum) <- c("iso3", "year", paste0(varname, "_", colnames(data2sum)[3:length(data2sum)]))
  return(data2sum)
}

##################  END OF FUNCTIONS ##################




#new based on full GADM 3.6 file, load full shapefile level 0
#Source: https://biogeo.ucdavis.edu/data/gadm3.6/gadm36_shp.zip
# from https://gadm.org/download_world.html
if(file.exists("gadm/gadm36_0.shp")){
gadm36_0_shp <- readOGR(dsn = "gadm/gadm36_0.shp")
}else{
  stop("Please download the shape file from https://biogeo.ucdavis.edu/data/gadm3.6/gadm36_shp.zip and place it in /gadm/gadm36_0.shp")
}


rcps = "historical"; if(dataset_name=="rcp_dasgupta") rcps = c("rcp26", "rcp45", "rcp60", "rcp85")

for(rcp in rcps){


if(dataset_name=="cru_1901_2016"){
  #Source: https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.01/
  dataset_folder = "V:\\CLIMATE\\UEA-CRU\\"  #1901-2016
  tmp <- brick(paste0(dataset_folder, "cru_ts4.01.1901.2016.tmp.dat.nc"), varname="tmp") # Mean monthly temperature
  pre <- brick(paste0(dataset_folder, "cru_ts4.01.1901.2016.pre.dat.nc"), varname="pre") #  Precipitation
  unit_tmp="celsius"; unit_pre="mm"; startyear = 1901; resolution = 0.5
}else if(dataset_name=="cru_2000s"){
  dataset_folder ="UEA-CRU\\"
  tmp <- brick(paste0(dataset_folder, "cru_ts4.01.2011.2016.tmp.dat.nc"), varname="tmp") # Mean monthly temperature
  pre <- brick(paste0(dataset_folder, "cru_ts4.01.2011.2016.pre.dat.nc"), varname="pre") #  Precipitation
  #just test
  plot(pre$X2011.01.16)
  #add previous decade(s)
  tmp2 <- brick(paste0(dataset_folder, "cru_ts4.01.2001.2010.tmp.dat.nc"), varname="tmp") # Mean monthly temperature
  pre2 <- brick(paste0(dataset_folder, "cru_ts4.01.2001.2010.pre.dat.nc"), varname="pre") #  Precipitation
  tmp <- raster::stack(tmp, tmp2)
  pre <- raster::stack(pre, pre2)
  unit_tmp="celsius"; unit_pre="mm"; startyear = 2001; resolution = 0.5
}else if(dataset_name=="rcp_dasgupta"){
  #Dasgupta RCPs: 2.5 degree data, monthly, 1861-2100, 170 countries!!!
  #Source: http://climexp.knmi.nl/selectfield_cmip5.cgi?id=someone@somewhere
  dataset_folder ="V:\\CLIMATE\\CMIP5_MODMEAN_DASGUPTA\\"
  dataset_folder ="CMIP5_MODMEAN_DASGUPTA/"
  tmp <- brick(paste0(dataset_folder, str_glue("tas_Amon_modmean_{rcp}_ave.nc")), varname="tas") #TAS
  pre <- brick(paste0(dataset_folder, str_glue("pr_Amon_modmean_{rcp}_ave.nc")), varname="pr") #  Precipitation
  #they need to be converted to -180, +180 (CRU-UEA is ok!!)
  pre <- rotate(pre)
  tmp <- rotate(tmp)
  #plus reduced to countries (exlude arctica, antarctica etc. if population weights are used)
  #if(pop2000weighted) extent(tmp) <- c(-180,180, -58,85); extent(pre) <- c(-180,180, -58,85)
  #just test
  unit_tmp="kelvin"; unit_pre="kg/m2/s"; startyear = 1861; resolution = 2.5
}else if(dataset_name=="spei"){
  # SPEI (Standardised Precipitation-Evapotranspiration Index (SPEI) from http://spei.csic.es/database.html)
  # Here: 12 month SPREI from http://digital.csic.es/handle/10261/153475 NetCDF files
  dataset_folder ="V:\\CLIMATE\\SPEI\\"
  tmp <- brick(paste0(dataset_folder, "spei_12m_1901_2015.nc"), varname="spei")
  unit_tmp="kelvin"; unit_pre="kg/m2/s"; startyear = 1901; resolution = 0.5
}else if(dataset_name=="HadCRUT5"){
  dataset_folder ="HadCRUT5\\"
  tmp <- brick(paste0(dataset_folder, "HadCRUT.5.0.1.0.analysis.anomalies.ensemble_mean.nc"), varname="tas_mean")
  unit_tmp="kelvin"; unit_pre="kg/m2/s"; startyear = 1901; resolution = 5
}

  #just to test
  plot(tmp[[1]])

  #get gridded population for population weighted averages
  #glp00ag30 contains 0.5 degree data, glp00ag on the other hand 2.5 arc minutes (0.041666 degrees)
  if(pop2000weighted){
    population_grid <- raster("gadm/gridded_population/glp00ag30.bil")
    #projection(population_grid) <- "+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0" # not sure if entirely necessary
    #e <- extent(0,360,-90,90) # match the existing raster
    #population_grid <- extend(population_grid, e) # pad a few values to reach all Latitudes
    #extent(population_grid) <- extent(tmp)
    #population_grid <- raster::aggregate(population_grid, fun=mean, fact=c(resolution/0.02777778, resolution/0.04166667)) #NOT SURE IF NEEDED
    population_grid <- raster::aggregate(population_grid, fun=mean, fact=c(resolution/0.5, resolution/0.5)) #NOT SURE IF NEEDED
    #population_grid
    #check overlap and coordinates
    plot(tmp[[1]])
    plot(population_grid[[1]], bg="transparent", add=TRUE)
  }



#get monthly iso3 dataset [TAKES LONG!!!!!!!!!!]
temperature_iso3_monthly <- raster2iso3(tmp, iso3 = iso3_save)
precipitation_iso3_monthly <- raster2iso3(pre, iso3 = iso3_save)



#convert to year iso3 datasets
temperature <- month2year(temperature_iso3_monthly, "temperature", startyear = startyear, unit=unit_tmp)
precipitation <- month2year(precipitation_iso3_monthly, "precipitation", startyear = startyear, unit=unit_pre)
climate_dataset_iso3_year <- merge(temperature, precipitation, by = c("iso3", "year"))

#clean dataset
print(str_glue("Climate dataset for years {min(climate_dataset_iso3_year$year)}-{max(climate_dataset_iso3_year$year)} and {length(unique(climate_dataset_iso3_year$iso3))} countries."))

if(pop2000weighted) save(climate_dataset_iso3_year, file = paste0(dataset_folder, str_glue("climate_dataset_iso3_year_{rcp}_pop2000weighted.Rdata")))
if(!pop2000weighted) save(climate_dataset_iso3_year, file = paste0(dataset_folder, str_glue("climate_dataset_iso3_year_{rcp}.Rdata")))


#case for SPEI
#spei12 <- temperature %>% dplyr::select(iso3, year, temperature_mean) %>% mutate(temperature_mean=temperature_mean+273.15) %>% dplyr::rename(spei12=temperature_mean)
#save(spei12, file = paste0(dataset_folder, str_glue("spei12.Rdata")))




}#close loop over rcps




if(iso3_save != "all" & dataset_name=="rcp_dasgupta"){

load(paste0(dataset_folder, "climate_dataset_iso3_year_rcp26_pop2000weighted.Rdata"))
climate_country <- climate_dataset_iso3_year %>% filter(iso3==iso3_save) %>% mutate(RCP="RCP26")
load(paste0(dataset_folder, "climate_dataset_iso3_year_rcp45_pop2000weighted.Rdata"))
climate_country <- rbind(climate_country, climate_dataset_iso3_year %>% filter(iso3==iso3_save) %>% mutate(RCP="RCP45"))
load(paste0(dataset_folder, "climate_dataset_iso3_year_rcp60_pop2000weighted.Rdata"))
climate_country <- rbind(climate_country, climate_dataset_iso3_year %>% filter(iso3==iso3_save) %>% mutate(RCP="RCP60"))
load(paste0(dataset_folder, "climate_dataset_iso3_year_rcp85_pop2000weighted.Rdata"))
climate_country <- rbind(climate_country, climate_dataset_iso3_year %>% filter(iso3==iso3_save) %>% mutate(RCP="RCP85"))
write.csv(climate_country, file = str_glue("climate_variables_historical_RCPs_{iso3_save}.csv"), row.names = F)
stop("Don't only take temp mean and compute moving average")
# <<<<< FUNCTION 2: Interpolation function (from mod_inequality in WITCH) >>>>>>
source('interpolate_long_dataset.R')


#compute 20 year moving average
climate_country <- interpolate_long_dataset(climate_country %>% select(iso3, year, temperature_mean, RCP) %>% dplyr::rename(value=temperature_mean) %>% mutate(variable="temperature_mean"), idvars = c("iso3", "RCP"), average_periods = 20)
write.csv(climate_country, file = str_glue("climate_variables_historical_RCPs_{iso3_save}.csv"), row.names = F)

}
