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

### Usage
python TEMPO_Raster_to_Statistics.py "/home/ghost/Obsidian Vault/Work/Research Position/data/scans/" "/home/ghost/Obsidian Vault/Work/Research Position/outputs/" O3PROF single 2012083123 2015083123 mean

##### Parameters:
- python geoTEMPOZip.py path_to_data_dir path_to_save_dir product mode start_date end_date stats_type
    - path_to_data_dir:
       - File path to location of granules to analyze

    - path_to_save_dir:
        - File path to save output csv file

    - product:
        - Product to analyze
        - Valid values
            - "O3PROF"

    - mode:
        - Mode to anaylze with
        - Valid values
           - "single"
           - "midday"
           - "whole"

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

### Methods
- Data from the TEMPO mission is recieved in a NetCDF format, which contains latitude and longitude points assciated with the level of a given product in Dobson units.
- The program recieves a directory containing the NetCDF files to analyze.  The data points along with the products levels from these files are combined into a GeoDataFrame from the GeoPandas python library.
- County or Census Tract shape files are imported into the program and converted into a GeoDataFrame containing an entry for each region in the shape files.
- The data and geographic GeoDataFrames are anaylzed for intersections, and a specified statistic is performed on the product values for each geographic region.
- Additionally, the latitude and longitude values of the points in a region are collected and stored in the resulting combined GeoDataFrame.
- Finally, the GeoDataFrame is converted to a CSV file and exported to the path specified.


### Example Output
![Example output](https://github.com/mewoocat/TEMPO_Raster_to_Statistics/blob/main/output.png)
