# TEMPO_Raster_to_Statistics

### Dependencies
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

Parameters:
- python geoTEMPOZip.py path_to_data_dir path_to_save_dir product mode start_date end_date stats_type
    - path_to_data_dir:
       - File path to location of granules to analyze

    - path_to_save_dir:
        - File path to save output csv file

    - product:
        - Product to analyze
        - Valid values
            - O3PROF

    - mode:
        - Mode to anaylze with
       Valid values
           single
           midday
           whole

   start_date:
       Date and time to start the analysis
       Format
           YYYYMMDDHH
   
   end_date:
       Date and time to end the analysis
       Format
           YYYYMMDDHH

   stats_type:
       Type of statistics to run
       Valid values
           mean
           max
           median
