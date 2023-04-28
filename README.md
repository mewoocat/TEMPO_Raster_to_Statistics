# TEMPO_Raster_to_Statistics

### Python Version
- Python 3.10.10

### Python Libraries Used
- netCDF4
- numpy
- sys
- matplotlib
- cartopy
- pandas
- glob
- geopandas
- shapely
- datetime
- pyresample

### Setup
- Install Python
- Install libraries used above
- Clone github repo
- Gain access to L2 or L3 TEMPO data

### Usage 
##### Parameters:
- python  geoTEMPOZip.py  path_to_data_dir  path_to_save_dir  product mode  start_date  end_date  stats_type
    
    - path_to_data_dir:
        - File path to location of granules to analyze

    - path_to_save_dir:
        - File path to save output csv file

    - product:
        - Product to analyze
        - Valid values
            - O3PROF

    - mode: (Warning: Not implemented, use any of the values)
        - Mode to anaylze with
        - Valid values
           - single
           - midday
           - whole

    - start_date:
        - Date and time to start the analysis
        - Format
            - YYYYMMDDHH
   
    - end_date:
        - Date and time to end the analysis
        - Format
        - YYYYMMDDHH

    - stats_type:
        - Type of statistics to run
        - Valid values
            - "mean"
            - "max"
            - "median"

##### Example
- ```python TEMPO_Raster_to_Statistics.py "/home/andrew/Obsidian Vault/Work/Research Position/data/scans/" "/home/andrew/Obsidian Vault/Work/Research Position/outputs/" O3PROF single 2012083123 2015083123 mean```

##### Note:
- This project is in an incomplete state, so unexpected issues may arise when running the program
- Testing has only occurred with O3PROF and NO2 products
- There are additional parameters located near the top of the file that must be set in order for the program to work properly
    - tempoFile
        - Smoothing functionality gains a single .nc file from this variable as opposed to reading in the directory specified.
    - smooth
        - True/False value for determining whether to run smoothing
    - geoMode
        - Either `counties` or `census_tract`
        - sets the region to analyze the data with
    - level
        - Either `2` or `3`
        - Set accordingly to input data level

### Features
- Read in a directory of .nc files at once
- Select a time frame of granules for level 2 data
- Process level 2 data with with or without smoothing
- Perform mean, median, and mode on data
- Compare data against counties and census_tracts 

##### Incomplete
- Process a single level 3 file with and without smoothing (Incomplete)
- Plotting of data (Incomplete)

### Methods
- Data from the TEMPO mission is received in a NetCDF format, which contains latitude and longitude points associated with the level of a given product in DU.
- The program recieves a directory containing the NetCDF files to analyze.  The data points along with the products levels from these files are combined into a GeoDataFrame from the GeoPandas python library.
- County or Census Tract shape files are imported into the program and converted into a GeoDataFrame containing an entry for each region in the shape files.
- The data and geographic GeoDataFrames are analyzed for intersections, and a specified statistic is performed on the product values for each geographic region.
- Additionally, the latitude and longitude values of the points in a region are collected and stored in the resulting combined GeoDataFrame.
- Finally, the GeoDataFrame is converted to a CSV file and exported to the path specified.

### Design overview
- Code sections
    - Variables
        - Additional parameters to adjust as needed
    - Counties
        - Creates GeoDataFrame from U.S. County shapefile
    - Main function
        - Get arguments
            - Handles arguments inputted into program
        - Get desired granules
            - Grabs L2 granules from the path and time frame specified
        - Generate GeoDataFrame from granules (Center point) 
            - Reads geographic and product data from granules into a GeoDataFrame
        - Generate GeoDataFrame from granules (Smoothing) 
            - Create polygons from L2 data to check for geographic boundaries overlaps (Does not function properly due to L2 data not being gridded)
            - Create polygons from L3 data to check for geographic boundaries overlaps (Incomplete)
        - Statistics
            - Perform selected statistic on GeoDataFrame of data associated with geographic boundaries
        - Outputs
            - Output resulting GeoDataFrame as comma delimited csv file
        - Plots (Incomplete/Non functional)
            - Prototypes of data plotting
        - Functions
            - Functions used for plotting and data retrieval

### What I learned
- Initially going into this project I had little to no experience working with geospatial data.  The beginning phases mostly consisted of me reading an existing script, and trying to grasp some of the concepts and libraries used.  I browsed through the docs for many of these to learn how to implement the features.  From there I began pulling bits and pieces from it to form the start of this script here.  A lot of brainstorming occurred at this point and I started to implement some of my ideas.  I began to realize that I was trying to reinvent the wheel in many cases and found libraries that handled these tasks more efficiently and effectively.  The core data management library in use here is GeoPandas.  Discovering this allowed for a convenient method of processing the TEMPO data into a tabular form and then performing statistics.  To compared the data against geographic boundaries I learned of and used shape files.  In this case, shape files for U.S. counties and census tracts were implemented.  At this point the way a pixel of data is compared with a geographic is by whether the center point of the pixel lies within the boundary.  However I learned that this approach can make the data much less useful in areas with small geographic boundaries in relation to the pixel size.  I implemented a sort of smoothing feature which checks whether any point in a pixel overlaps with a geographic boundary.  The intent was to add weights to the data that would adjust the amount by how much area of overlap exists.  However, this has yet to be implemented. Overall, working on this project has been a very educational and fascinating experience.  I wish to have developed the project further, but I am relatively satisfied with the progress I have made.


### Output
- Output is in the form of a comma delimited csv file.

##### Example output
![Example output](https://github.com/mewoocat/TEMPO_Raster_to_Statistics/blob/main/output.png)
