### Readme ###
#
# Example usage:
#       python TEMPO_Raster_to_Statistics.py "/home/ghost/Downloads/data/scans/" "/home/ghost/obsidian_vault/Work/Research Position/outputs/" O3PROF single 2012083123 2015083123 mean

# Parameters:
#   python geoTEMPOZip.py path_to_data_dir path_to_save_dir product mode start_date end_date stats_type
#   
#   path_to_data_dir:
#       File path to location of granules to analyze
#
#   path_to_save_dir:
#       File path to save output csv file
#
#   product:
#       Product to analyze
#       Valid values
#           O3PROF
#
#   mode: (Warning: Not implemented)
#       Mode to anaylze with
#       Valid values
#           single
#           midday
#           whole
#
#   start_date:
#       Date and time to start the analysis
#       Format
#           YYYYMMDDHH
#   
#   end_date:
#       Date and time to end the analysis
#       Format
#           YYYYMMDDHH
#
#   stats_type:
#       Type of statistics to run
#       Valid values
#           mean
#           max
#           median


### Imports ###
from netCDF4 import Dataset
import numpy as np
import sys
import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
#import cartopy.io.shapereader as shpreader
import pandas as pd
import glob
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import Polygon
from pyresample import geometry
from datetime import datetime
#from numba import jit


### Variables ###
# Data file
#tempoFile = "/home/ghost/Downloads/data/TEMPO_O3PROF-PROXY_L2_V01_20130831T212359Z_S013G05.nc"
tempoFile = "/home/ghost/Downloads/data/L3/TEMPO_NO2-PROXY_L3_V01_20130714T180000Z_S010.nc"
#tempoFile = "C:/Users/ghost/Downloads/TEMPO_NO2-PROXY_L3_V01_20130714T180000Z_S010.nc"
#product = "O3PROF"
#product = "NO2"
coltype = 'trop'
cldthresh = 0.3
smooth = False
# county or census_tract
geoMode = "counties"
level = 2

##################################################################
### counties
##################################################################

# Read in county shape file as geoDataFrame
counties = gpd.read_file('assets/cb_2018_us_county_500k/cb_2018_us_county_500k.shp')


##################################################################
### Census tracts
##################################################################
def constructCensusTracts():
    print("Constructing cenus track data structure...")
    census_tract_dir = "assets/census_tracts/"
    tract_folders = np.sort(glob.glob(census_tract_dir + '*/'))
    # Array of census tract shape files as GeoDataFrames
    tracts = []
    for tract_folder in tract_folders:
        tract_shp = glob.glob(tract_folder + '*.shp')
        tract_GDF = gpd.read_file(tract_shp[0])
        tracts.append(tract_GDF)

    tractCombined = pd.concat(tracts)

    print("Finished constructing cenus track data structure.")

    #pd.set_option('display.max_rows', None)
    #print(tractCombined)
    return tractCombined

#print(tractCombined)



if geoMode == "counties":
    geoContext = counties
elif geoMode == "census_tract":
    tractCombined = constructCensusTracts()
    geoContext = tractCombined
else:
    geoContext = "We have a problem"

### Main function ###
def main():
    
    ##################################################################
    ### Get arguments ###
    ##################################################################

    # Location of data files (Path to folder)
    dataDir = sys.argv[1]

    # Location to save output (Path to folder)
    saveDir = sys.argv[2]

    # Product to analyze ("O3PROF", )
    product = sys.argv[3]

    # Time frame of analysis ("single", "midday", "whole")
    timeFrame = sys.argv[4]

    # Start date and time (YYYYMMDD)
    startDate = sys.argv[5]
    startDate = datetime.strptime(startDate, "%Y%m%d%H")

    # End date and time (YYYYMMDD)
    endDate = sys.argv[6]
    endDate = datetime.strptime(endDate, "%Y%m%d%H")

    # Type of statistics to perform ("mean", "max")
    statsType = sys.argv[7]
    

    ##################################################################
    ### Get desired granules
    ##################################################################

    granules = np.sort(glob.glob(dataDir + "*.nc" ))
    sortedGranules = []
    print(granules)
    for gran in granules:
        #gran = maindir+'/'+product+'/'+'TEMPO_'+iprod+'-PROXY'+'_L2_V01_'++'T'+ifile[8:]+'*Z_*G??'+'.nc'
        print(gran)

        # Get date in file name
        granDate = gran.split("_")[4]
        granDateYYYYMMDD = granDate.split("T")[0]
        granDateHH = granDate.split("T")[1][0:2]
        granDate  = datetime.strptime(granDateYYYYMMDD + granDateHH, "%Y%m%d%H")
        print(granDate)

        # Get product for current granule
        granProduct = gran.split("_")[1].split("-")[0]
        print(granProduct)

        # Only add granules that fall within the date/time range and are the desired product    file=open(saveDir + "lat_bounds.txt", "w+")
        if (granDate >= startDate and granDate <= endDate and granProduct == product):
            sortedGranules.append(gran)
        

    if not smooth:
        ################################################################## 
        ### Generate GeoDataFrame from granules (Center point) 
        ################################################################## 

        print("Starting point calculations...")

        GeoDataFrames_granules = []
        for gran in sortedGranules:
            print("Processing: " + gran)

            # Get geographic info for each data point
            geoData = get_latlon(gran, product, "center")
            lat = geoData['lat']
            lon = geoData['lon']
            
            # Issue: changes decimal representation slightly
            # Convert 2D latitude array to 1D array
            pixel_lat = np.array([])
            for i in range(len(lat)):
                pixel_lat = np.concatenate((pixel_lat, lat[i]))

            # Convert 2D longitude array to 1D array
            pixel_lon = np.array([])
            for i in range(len(lon)):
                pixel_lon = np.concatenate((pixel_lon, lon[i]))

            # Gets variable data
            varData = get_vardata(gran, product)
            var = varData['data']
            long_name = varData['long_name']

            pixel_var = np.array([])
            for i in range(len(var)):
                # For some reason the pixel_var array doesn't have commas
                pixel_var = np.concatenate((pixel_var, var[i]))

            dataFrame = pd.DataFrame({'lon':pixel_lon, 'lat':pixel_lat, 'varP':pixel_var})
            #print(dataFrame)
     
            GeoDataFrames_granules.append(dataFrame)


        # Combine granule dataframes into a single dataframe
        combinedGransDF = pd.concat(GeoDataFrames_granules)

        # Convert combined dataframe into a geodataframe
        combinedGransGDF = gpd.GeoDataFrame(combinedGransDF, geometry=gpd.points_from_xy(combinedGransDF.lon, combinedGransDF.lat), crs=geoContext.crs)
        
        # Create geodataframe sorted by county
        # Doesn't need the tools portion?
        combinedGransGDFInCounties = gpd.tools.sjoin(combinedGransGDF, geoContext, predicate="within", how="left")
        # Drop rows that don't have a name or var value
        combinedGransGDFInCounties = combinedGransGDFInCounties.dropna(subset=['NAME'])
        combinedGransGDFInCounties = combinedGransGDFInCounties.dropna(subset=['varP'])
        
        # Convert variable units (Very terrible estimate, don't use!)
        # 2.69matplotlib.patches DU equals 0.001 ppm  From: https://www.ablison.com/what-is-a-dobson-unit-du/
        # 2690 DU equals 1 ppm 
        #combinedGransGDFInCounties["varP_PPM"] = combinedGransGDFInCounties['varP'] / 2690
        
        # Convert variable units from DU to ppb for 0-2 km ozone product
        # Method:
        # 1 DU = 2.69 x 10^16 molecules / cm2; Use this to convert from DU to molecules / cm2.
        # 2 Then using our knowledge of the 0-2 km layer, we can divide the result from (1) by 2 km (or 200000 cm) to get a result in molecules / cm3
        # 3Next, use the relation on the cheat sheet here (https://cires1.colorado.edu/jimenez-group/Press/2015.05.22-Atmospheric.Chemistry.Cheat.Sheet.pdf) to convert to ppb.  Relation 1 ppb = 2.46 x 10^10 molecules / cm3. 

        combinedGransGDFInCounties["varP_PPB"] = ((combinedGransGDFInCounties['varP'] * (2.69 * (10 ** 16))) / 200000) / (2.46 * (10 ** 10))


        # Raw data points without statistics
        raw = combinedGransGDFInCounties

        GDF = combinedGransGDFInCounties

        print("Finished point calculations...")


    if smooth:
        ##################################################################
        ### Generate GeoDataFrame from granules (Smoothing) 
        ##################################################################

        # If data is level 2
        if level == 2:
            print("Starting smoothing calculations...")
            
            # Get pixel bounds
            geoData = get_latlon(tempoFile, "O3PROF", "corner")
            lat_bounds = geoData['lat']
            lon_bounds = geoData['lon']

            # 62976 is from 123 * 512
            # 4 is each corner
            pixel_lat_bounds = np.reshape(lat_bounds, (62976, 4))
            pixel_lon_bounds = np.reshape(lon_bounds, (62976, 4))

            pixel_lat_bounds = np.reshape(pixel_lat_bounds, (62976, 4, 1))
            pixel_lon_bounds = np.reshape(pixel_lon_bounds, (62976, 4, 1))

            # Combine lon and lat into same array
            polygon_bounds = np.concatenate((pixel_lon_bounds, pixel_lat_bounds), axis=2)

            # Swap points 2 and 3 so the polygon renders correctly
            polygon_bounds[:, [1,2], :] = polygon_bounds[:, [2,1], :]

            # Convert polygon_bounds into a list of polygons
            pixel_polygons = np.array([])
            for i in range(len(polygon_bounds)):
                # Polygon rounds to 3 decimal
                pixel_polygons=np.append(pixel_polygons,Polygon(polygon_bounds[i]))
               
            # Associate pixel polygons with product variables
            polygonVar = get_vardata(tempoFile, product)
            polygonVar = polygonVar['data']
            polygonVar2 = np.array([])
            for i in range(len(polygonVar)):
                polygonVar2 = np.concatenate((polygonVar2, polygonVar[i]))
            polygonVar = polygonVar2

            # Create GeoDataFrame of pixel polygons
            polygonDF = pd.DataFrame({'varP':polygonVar})
            polygonGDF = gpd.GeoDataFrame(polygonDF, geometry=pixel_polygons, crs=geoContext.crs)

            # Find pixel polygon with county intersections
            intersectionGDF = gpd.overlay(polygonGDF, geoContext, how='union')
            intersectionGDF = intersectionGDF.dropna(subset=['NAME'])
            intersectionGDF = intersectionGDF.dropna(subset=['varP'])

            # Convert DU to PPM
            #intersectionGDF["varP_PPM"] = intersectionGDF['varP'] / 2690
            
            # Convert DU to PPB
            intersectionGDF["varP_PPB"] = ((intersectionGDF['varP'] * (2.69 * (10 ** 16))) / 200000) / (2.46 * (10 ** 10))
            
            GDF = intersectionGDF

            print("Finish intersection calculations.")

        # If data is level 3
        if level == 3:
            print("Starting smoothing calculations...")
            
            # Get pixel bounds
            geoData = get_latlon(tempoFile, "O3PROF", "center")
            lat = geoData['lat']
            lon = geoData['lon']

            gridCenterOffset = 0.025 
           
            # Resize lat and lon values
            gridLon = np.resize(lon, (len(lat), len(lon)))
            gridLat = np.resize(lat, (len(lon), len(lat)))
            gridLat = np.transpose(gridLat)
            print(np.shape(gridLon))
            print(np.shape(gridLat))

            
            # Experimenting with creating a grid of lat lon values 
            grid = geometry.GridDefinition(lons=gridLon, lats=gridLat)
            print(grid)

            gridLon = np.resize(gridLon, (2454340))
            gridLat = np.resize(gridLat, (2454340))
            print(gridLon)
            print(gridLat)
            print(np.shape(gridLon))
            print(np.shape(gridLat))
            
            
            # Create an array of polygons of the pixels using the center point and an offset
            # Warning, very inefficient, takes several hours to run on my machine
            """
            grid_polygons = np.array([])
            for i in (range(len(gridLon))):
                print(i)
                nw = (gridLon[i] - gridCenterOffset, gridLat[i] + gridCenterOffset)
                ne = (gridLon[i] + gridCenterOffset, gridLat[i] + gridCenterOffset) 
                se = (gridLon[i] + gridCenterOffset, gridLat[i] - gridCenterOffset)
                sw = (gridLon[i] - gridCenterOffset, gridLat[i] - gridCenterOffset) 
                cornerPoints = [nw, ne, se, sw]
                #print(sw)
                #print(cornerPoints)
                grid_polygons = np.append(grid_polygons, Polygon(cornerPoints))
            """


            # Experimenting with code from Aaron
            """ 
            ### Latest grid specifications for TEMPO L3 products as of 12/2022
            ## 0.05 degree grid spacing
            minlat, maxlat = 17.025, 63.975
            minlon, maxlon = -154.975, -24.475
            gridres = 0.05

            flats = np.linspace(minlat,maxlat,num=int(((maxlat-minlat)/gridres)+1),endpoint=True)
            flons = np.linspace(minlon,maxlon,num=int(((maxlon-minlon)/gridres)+1),endpoint=True)

            numlats = len(flats)
            numlons = len(flons)

            gridlon = np.resize(flons,(numlats,numlons))
            gridlat = np.resize(flats,(numlons,numlats))
            gridlat = np.transpose(gridlat)

            grid_fixed = geometry.GridDefinition(lons=gridlon,lats=gridlat)

            ### Grid for pcolormesh plotting #################
            flats_d = np.linspace(int(minlat-(gridres/2)),int(maxlat+(gridres/2)),num=int(((maxlat-minlat)/gridres)+2),endpoint=True)
            flons_d = np.linspace(int(minlon-(gridres/2)),int(maxlon+(gridres/2)),num=int(((maxlon-minlon)/gridres)+2),endpoint=True)

            print(flats_d)

            #flats = np.linspace(beglat-(gridres/2),endlat-(gridres/2),num=((endlat-beglat)/gridres)+1,endpoint=True)
            #flons = np.linspace(beglon+(gridres/2),endlon+(gridres/2),num=((endlon-beglon)/gridres)+1,endpoint=True)

            numlats = len(flats_d)
            numlons = len(flons_d)

            gridlon_mesh = np.resize(flons_d,(numlats,numlons))
            gridlat_mesh = np.resize(flats_d,(numlons,numlats))
            gridlat_mesh = np.transpose(gridlat_mesh)
            
            gridlon_mesh = np.resize(gridlon_mesh, (2454340))
            gridlat_mesh = np.resize(gridlat_mesh, (2454340))

            print(gridlon_mesh)
            print(gridlat_mesh)
            print(np.shape(gridlon_mesh))
            print(np.shape(gridlat_mesh))
            """

             
            # Associate pixel polygons with product variables
            polygonVar = get_vardata(tempoFile, product)
            polygonVar = polygonVar['data']
            polygonVar1D = np.array([])
            for i in range(len(polygonVar)):
                polygonVar1D = np.concatenate((polygonVar1D, polygonVar[i]))

            # Create GeoDataFrame of pixel polygons
            polygonDF = pd.DataFrame({'varP':polygonVar1D})
            print(polygonDF)
            polygonGDF = gpd.GeoDataFrame(polygonDF, geometry=grid_polygons, crs=geoContext.crs)

            # Find pixel polygon with county intersections
            intersectionGDF = gpd.overlay(polygonGDF, geoContext, how='union')
            intersectionGDF = intersectionGDF.dropna(subset=['NAME'])
            intersectionGDF = intersectionGDF.dropna(subset=['varP'])

            # Convert DU to PPM
            #intersectionGDF["varP_PPM"] = intersectionGDF['varP'] / 2690
            
            # Convert DU to PPB
            intersectionGDF["varP_PPB"] = ((intersectionGDF['varP'] * (2.69 * (10 ** 16))) / 200000) / (2.46 * (10 ** 10))
            
            GDF = intersectionGDF

            GDF = counties

            print("Finish intersection calculations.")


    ##################################################################
    ### Statistics
    ##################################################################

    
    # Use dissolve instead of groupby?
    # https://geopandas.org/en/stable/docs/user_guide/aggregation_with_dissolve.html

    print(GDF.head())

    # Checks for statistic type
    if (statsType == "mean"):
        print("Calculating mean...")
        # Group rows by county and calculate mean of variable for each county
        #GDF = GDF.groupby(['STATEFP', 'COUNTYFP', 'NAME'])['varP', 'varP_PPM'].mean()
        GDF = GDF.dissolve(by=['STATEFP', 'COUNTYFP', 'NAME'], aggfunc='mean')

    if (statsType == "max"):
        print("Calculating max...")
        # Group rows by county and calculate max of variable for each county
        #GDF = GDF.groupby(['STATEFP', 'COUNTYFP', 'NAME'])['varP', 'varP_PPM'].max()
        GDF = GDF.dissolve(by=['STATEFP', 'COUNTYFP', 'NAME'], aggfunc='mean')
        
    if (statsType == "median"):
        print("Calculating max...")
        # Group rows by county and calculate max of variable for each county
        #GDF = GDF.groupby(['STATEFP', 'COUNTYFP', 'NAME'])['varP', 'varP_PPM'].max()
        GDF = GDF.dissolve(by=['STATEFP', 'COUNTYFP', 'NAME'], aggfunc='median')

    print(GDF.head())

    
    ##################################################################
    ### Outputs
    ##################################################################
    
    currentTime = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    # Output data as csv
    #raw.to_csv(saveDir + "raw_out.csv")
    #polygonGDF.to_csv(saveDir + "polygons.csv")
    GDF.to_csv(saveDir + "output__" + statsType + currentTime +".csv")

    print(f"Saved output to {saveDir}")

    
    ##################################################################
    ### Plots
    ##################################################################
    
    # Plot Pixel polygons
    if smooth:
        print("Plot...")
        #plot(piexl_polygons)
        #plot2(polygonGDF)


##################################################################
### Functions
##################################################################

def plot2(polygons):
    ax = counties.plot(color='green', edgecolor='black')
    polygons.plot(ax=ax, color='red', edgecolor='blue', alpha=0.5)
    plt.show()

def plot(polygons):
    
    print("Starting plot...")

    shpdir = "/home/ghost/Obsidian Vault/Work/Research Position/data/cb_2018_us_county_500k/cb_2018_us_county_500k.shp"
    reader = shpreader.Reader(shpdir)
    counties = list(reader.geometries())
    COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

    linewidth = 1.0

    # Set projection type?
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Set lat lon bounds
    ax.set_extent([-150,-45,0,80], ccrs.PlateCarree())

    #ax.coastlines(resolution='10m',linewidth=1.0)
    ax.coastlines()

    ax.add_feature(cfeature.NaturalEarthFeature(scale='10m',edgecolor='black',\
        category='cultural',name='admin_0_countries',facecolor='none',linewidth=1.0))

    ax.add_feature(COUNTIES, facecolor='none', edgecolor='black',linewidth=linewidth)


    gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True, linewidth=0.5, color='black', alpha=0.7, linestyle='--')

    # Adding the title
    plt.title("Map")
    
    """ 
    for p in polygons:
        #ax.add_patch(p)
        plt.plot(p.exterior.xy)
    """
    #p = Polygon([[-80, 20], [-90, 40], [-65, 45]])
    #plt.plot(p.exterior.xy)

    """
    # Add data to the plot
    colors = np.random.rand(62976)
    plt.scatter(
        x=pixel_lon,
        y=pixel_lat,
        #color="green",
        c=colors,
        s=8,
        alpha=1,
        transform=ccrs.PlateCarree()
    )
    """

    plt.show()

# Get latitude and longitude from .nc file
def get_latlon(data,iprod, type):
    print ('***** PROCESSING FILE ******* ')
    print(data)

    fill = -1.267651e+30

    #print (data)
    ncfile = Dataset(data,'r')
 # Read geolocation group
    geo =  ncfile.groups['geolocation']
    #print("geo = ")
    #print(geo)

    if(type == "center" and level == 2):
    # if iprod != 'O3PROF':
    # For pcolormesh plot, read SW lat/lon bounds
        lat = geo.variables['latitude']
        fill = lat._FillValue
        #lat = lat[:,:,0]
        lat = lat[:,:]

        lat[np.isnan(lat)] = fill

        lon = geo.variables['longitude']
        fill = lon._FillValue
        #lon = lon[:,:,0]
        lon = lon[:,:]

        lon[np.isnan(lon)] = fill

    if(type == "center" and level == 3):
    # if iprod != 'O3PROF':
    # For pcolormesh plot, read SW lat/lon bounds
        lat = ncfile.variables['latitude']
        lat = lat[:]
        lat[np.isnan(lat)] = fill

        lon = ncfile.variables['longitude']
        lon = lon[:]
        lon[np.isnan(lon)] = fill

    if(type == "corner"):
    # if iprod != 'O3PROF':
    # For pcolormesh plot, read SW lat/lon bounds
        lat = geo.variables['latitude_bounds']
        fill = lat._FillValue
        lat = lat[:,:,:]
        #lat = lat[:,:] # ???

        lat[np.isnan(lat)] = fill

        lon = geo.variables['longitude_bounds']
        fill = lon._FillValue
        lon = lon[:,:,:]
        #lon = lon[:,:] # ???

        lon[np.isnan(lon)] = fill

  # Process SZA for QC
    sza = geo.variables['solar_zenith_angle']
    fill = sza._FillValue
    sza = sza[:,:]

    sza[np.isnan(sza)] = fill

    ncfile.close()

    return {'lat':lat,'lon':lon,'sza':sza}


# Get product data from .nc file
def get_vardata(data,iprod):
    print ('***** VALID FILE ******* ')
    ncfile = Dataset(data,'r')
# Read group and variable based on input argument
    if level == 2:
        vars = ncfile.groups['product']
        support = ncfile.groups['support_data']
        if iprod == 'HCHO' or iprod == 'SO2' or iprod == 'H2O':
            varget = vars.variables['vertical_column']
            fill = varget._FillValue
        if iprod == 'NO2':
            if coltype == 'trop' or coltype == 'total':
                varget = vars.variables['vertical_column_troposphere']
            if coltype == 'strat':
                varget = vars.variables['vertical_column_stratosphere']
            fill = varget._FillValue
        if iprod == 'O3PROF':
            if coltype == 'total':
                varget = vars.variables['total_ozone_column']
            if coltype == 'pbl':
                varget = vars.variables['ozone_profile']
            if coltype == 'trop':
                varget = vars.variables['troposphere_ozone_column']
            if coltype == 'strat':
                varget = vars.variables['stratosphere_ozone_column']
            fill = varget._FillValue
    if level == 3:
        vars = ncfile.groups['product']
        support = ncfile.groups['support_data']
        if iprod == 'HCHO' or iprod == 'SO2' or iprod == 'H2O':
            varget = vars.variables['vertical_column']
            fill = varget._FillValue
        if iprod == 'NO2':
            if coltype == 'trop' or coltype == 'total':
                varget = vars.variables['vertical_column_troposphere']
            if coltype == 'strat':
                varget = vars.variables['vertical_column_stratosphere']
            fill = varget._FillValue
        if iprod == 'O3PROF':
            if coltype == 'total':
                varget = vars.variables['total_ozone_column']
            if coltype == 'pbl':
                varget = vars.variables['ozone_profile']
            if coltype == 'trop':
                varget = vars.variables['troposphere_ozone_column']
            if coltype == 'strat':
                varget = vars.variables['stratosphere_ozone_column']
            fill = varget._FillValue


# Numpy array
    var = varget[:]
 # Replace masked (fill) value with NaN
    var = np.ma.filled(var,fill_value=np.nan)   

    if coltype == 'pbl':
        var = var[:,:,np.shape(var)[2]-1]

    ########### start quality control procedures ################
    if iprod != 'O3PROF':
    # Data quality for QC
        data_qc = vars.variables['main_data_quality_flag']#[:]
        fill = data_qc._FillValue
        data_qc = data_qc[:]

    # Data quality (good = 0, suspect = 1 (not used), bad = 2)
      # V01 only flags invalid data
        bad = np.where(data_qc > 0)
    # set flag field for masking poor data
        flags = (var*0)+0
    # Data quality (good = 0, suspect = 1 (not used), bad = 2)
      # V01 only flags invalid data
        bad = np.where(data_qc > 0)
        flags[bad] = 1

        data_qc = np.ma.filled(data_qc,fill_value=1)

        # Disregard poor data
        var[flags == 1] = np.nan

    # Cloud fraction for QC
        cldfra = support.variables['amf_cloud_fraction']#[:]
        cfill = cldfra._FillValue
        cldfra = cldfra[:]

    # Cloud fraction
        cldfra = np.ma.filled(cldfra,fill_value=1.0)
        bad = np.where((cldfra > cldthresh))

        var[bad] = np.nan

    # Get name attribute for plotting purposes later
        long_name = varget.long_name

    if iprod == 'O3PROF':
    # Cloud fraction for QC
        cldfra = support.variables['eff_cloud_fraction']
        cfill = cldfra._FillValue

        if coltype == 'pbl':
        # Get name attribute for plotting purposes later
            long_name = '0-2 km ozone'
        else:
            long_name = varget.comment

        cldfra = cldfra[:]

    # Cloud fraction
        cldfra = np.ma.filled(cldfra,fill_value=1.0)
        bad = np.where((cldfra > cldthresh))

        var[bad] = np.nan
## remove for now
##        var[var < -9.] = np.nan

#        cldfra = support.variables['ozone_averaging_kernel']
#        cfill = cldfra._FillValue
#        cldfra = cldfra[:]
#        print(np.nanmin(cldfra),np.nanmax(cldfra))
#        print(cldfra[100,300,:,0])
#        print('******************************************')
#        print('******************************************')
#        print('******************************************')

#        cldfra = support.variables['ozone_noise_correlation_matrix']
#        cfill = cldfra._FillValue
#        cldfra = cldfra[:]
#        print(np.nanmin(cldfra),np.nanmax(cldfra))
#        print(cldfra[100,300,:,0])
#        exit()

 ## Add section for calculating total column NO2
    if iprod == 'NO2':
        if coltype == 'total':
            varget2 = vars.variables['vertical_column_stratosphere'][:]
            var = var+varget2
    # Get name attribute for plotting purposes later
            long_name = 'Total'
 #########################################

    ########### end quality control procedures ################
    ncfile.close()

    return {'data':var,'long_name':long_name}


# Debug function for writing array out to file
def arrayToFile(array, filename):
    saveDir = "/home/ghost/Obsidian Vault/Work/Research Position/outputs/"
    file=open(saveDir + filename, "w+")
    file.write(str(array))
    file.close()


# Testing
def testing():

    print("@ testing...")


    sys.path.insert(0,'/home/ghost/Obsidian Vault/Work/Research Position/TEMPO Proxy Codes/')
    #import colormap_generator
    #import colormaps as cmaps


    gridres = 0.05

    ### Latest grid specifications for TEMPO L3 products as of 12/2022
    ## 0.05 degree grid spacing
    minlat, maxlat = 17.025, 63.975
    minlon, maxlon = -154.975, -24.475


    flats = np.linspace(minlat,maxlat,num=int(((maxlat-minlat)/gridres)+1),endpoint=True)
    flons = np.linspace(minlon,maxlon,num=int(((maxlon-minlon)/gridres)+1),endpoint=True)

    print("flats")
    print(flats)
    print("flons")
    print(flons)
     

    numlats = len(flats)
    numlons = len(flons)


    gridlon = np.resize(flons,(numlats,numlons))
    gridlat = np.resize(flats,(numlons,numlats))
    gridlat = np.transpose(gridlat)

    print("gridlon")
    print(gridlon)
    print(np.shape(gridlon))


    grid_fixed = geometry.GridDefinition(lons=gridlon,lats=gridlat)

    print("grid_fixed")
    print(grid_fixed)


    ### Grid for pcolormesh plotting #################
    flats_d = np.linspace(minlat-(gridres/2),maxlat+(gridres/2),num=int(((maxlat-minlat)/gridres)+2),endpoint=True)
    flons_d = np.linspace(minlon-(gridres/2),maxlon+(gridres/2),num=int(((maxlon-minlon)/gridres)+2),endpoint=True)

    print("flats_d")
    print(flats_d)
    print("flons_d")
    print(flons_d)


    #flats = np.linspace(beglat-(gridres/2),endlat-(gridres/2),num=((endlat-beglat)/gridres)+1,endpoint=True)
    #flons = np.linspace(beglon+(gridres/2),endlon+(gridres/2),num=((endlon-beglon)/gridres)+1,endpoint=True)
     

    numlats = len(flats_d)
    numlons = len(flons_d)
     

    gridlon_mesh = np.resize(flons_d,(numlats,numlons))
    gridlat_mesh = np.resize(flats_d,(numlons,numlats))
    gridlat_mesh = np.transpose(gridlat_mesh)

    print(gridlat_mesh)


##################################################################
### Run main function
##################################################################

main()
#testing()
