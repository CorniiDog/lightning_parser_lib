####################################################################################
#
# About: A top-down view of what is going on
#
####################################################################################
"""
This program processes LYLOUT data files, such as "LYLOUT_20220712_pol.exported.dat"

1. It first reads through all the files, and then puts all of the points into an 
SQLite database

2. Then, with user-specified filters, the user extracts a pandas DataFrame 
(DataFrame "events") from the SQLite database that meets all of the 
filter criteria. 

3. Afterwards, with user-specified parameters, the lightning_bucketer processes 
all of the "events" data to return a list of lightning strikes, which each 
lightning strike is simply a list of indices for the "events" DataFrame
(a list of lists).

4. You can use the events with the lightning strikes data to plot data or analyze 
the data. Examples in the code and comments below show how to do so.
"""
####################################################################################
print("Starting up. Importing...")
import lightning_parser_lib.config_and_parser as config_and_parser
from lightning_parser_lib.number_crunchers.toolbox import tprint
import lightning_parser_lib.number_crunchers.toolbox as toolbox
from lightning_parser_lib.number_crunchers.lightning_visualization import XLMAParams
import time
import datetime
import pandas as pd

# what percent of the total number of cores to be utilized. 
# Set to 0.0 to use only one core
CPU_PCT = 0.9 

lightning_configuration = config_and_parser.LightningConfig(
    num_cores = toolbox.cpu_pct_to_cores(CPU_PCT),
    lightning_data_folder = "lylout_files",
    data_extension = ".dat",
    cache_dir ="cache_dir",
    csv_dir = "strikes_csv_files",
    export_dir = "export",
    strike_dir = "strikes",
    strike_stitchings_dir = "strike_stitchings"
)

EXPORT_AS_CSV = True 
EXPORT_GENERAL_STATS = True
EXPORT_ALL_STRIKES = False
EXPORT_ALL_STRIKES_STITCHINGS = False

config_and_parser.lightning_bucketer.USE_CACHE = True

def main():

    # Column/Header descriptions:
    # 'time_unix'    -> float   Seconds (Unix timestamp, UTC)
    # 'lat'          -> float   Degrees (WGS84 latitude)
    # 'lon'          -> float   Degrees (WGS84 longitude)
    # 'alt'          -> float   Meters (Altitude above sea level)
    # 'reduced_chi2' -> float   Reduced chi-square goodness-of-fit metric
    # 'num_stations' -> int     Count (Number of contributing stations)
    # 'power_db'     -> float   Decibels (dBW) (Power of the detected event in decibel-watts)
    # 'power'        -> float   Watts (Linear power, converted from power_db using 10^(power_db / 10))
    # 'mask'         -> str     Hexadecimal bitmask (Indicates contributing stations)
    # 'stations'     -> str     Comma-separated string (Decoded station names from the mask)
    # 'x'            -> float   Meters (ECEF X-coordinate in WGS84)
    # 'y'            -> float   Meters (ECEF Y-coordinate in WGS84)
    # 'z'            -> float   Meters (ECEF Z-coordinate in WGS84)
    # `file_name`    -> str     The name of the file used that contains the point information

    # Mark process start time
    process_start_time = time.time()

    ####################################################################################
    # Filter params for extracting data points from the SQLite database
    ####################################################################################
    start_time = datetime.datetime(2020, 4, 29, 13, 0, tzinfo=datetime.timezone.utc).timestamp()  # Timestamp converts to unix (float)
    end_time = datetime.datetime(2020, 4, 29, 14, 59, tzinfo=datetime.timezone.utc).timestamp()  # Timestamp converts to unix (float)

    # Build filter list for time_unix boundaries.
    # Look at "List of headers" above for additional
    # Filterings
    filters = [
        ("time_unix", ">=", start_time),  # In unix
        ("time_unix", "<=", end_time),  # In unix
        ("reduced_chi2", "<", 5.0,),  # The chi^2 (reliability index) value to accept the data
        ("num_stations", ">=", 5),  # Number of stations that have visibly seen the strike
        ("alt", "<=", 24000),  # alt is in meters. Therefore 20 km = 20000m
        ("alt", ">", 0),  # Above ground
        ("power_db", ">", -4),  # In dBW
        ("power_db", "<", 50),  # In dBW
    ]
    events: pd.DataFrame = config_and_parser.get_events(filters, config=lightning_configuration)
    tprint("Events:", events)

    ####################################################################################
    # Identifying the lightning strikes
    ####################################################################################

    # Additional parameters that determines "What points make up a single lightning strike"
    # They are explicitly defined
    params = {
        # Creating an initial lightning strike
        "max_lightning_dist": 30000,  # Max distance between two points to determine it being involved in the same strike
        "max_lightning_speed": 1.4e8,  # Max speed between two points in m/s (essentially dx/dt)
        "min_lightning_speed": 0,  # Min speed between two points in m/s (essentially dx/dt)
        "min_lightning_points": 100,  # The minimum number of points to pass the system as a "lightning strike"
        "max_lightning_time_threshold": 0.3,  # Max number of seconds between points 
        "max_lightning_duration": 30, # Max seconds that define an entire lightning strike. This is essentially a "time window" for all of the points to fill the region that determines a "lightning strike"

        # Caching
        "cache_results": True, # Set to true to cache results
        "max_cache_life_days": 7 # The number of days to save a cache
    }
    bucketed_strikes_indices, bucketed_lightning_correlations = config_and_parser.bucket_dataframe_lightnings(events, config=lightning_configuration, params=params)

    # Example: To get a Pandas DataFrame of the first strike in the list, you do:
    # ```
    # first_strikes = events.iloc[bucketed_strikes_indices[0]]
    # ```
    #
    # Example 2: Iterating through all lightning strikes:
    # ```
    # for i in range(len(bucketed_strikes_indices)):
    #   sub_strike = events.iloc[bucketed_strikes_indices[i]]
    #   # Process the dataframe however you please of the designated lightning strike
    # ```

    process_time = time.time() - process_start_time
    tprint(f"Process time: {process_time:.2f} seconds.")
    config_and_parser.display_stats(events, bucketed_strikes_indices)

    ####################################################################################
    # Plotting and exporting
    ####################################################################################

    # Only export plot data with more than n datapoints
    MAX_N_PTS = 1000
    bucketed_strikes_indices, bucketed_lightning_correlations = config_and_parser.limit_to_n_points(bucketed_strikes_indices, bucketed_lightning_correlations, MAX_N_PTS)

    if EXPORT_AS_CSV:
        config_and_parser.export_as_csv(bucketed_strikes_indices, events, config=lightning_configuration) 

    # Add a zipped file for counties into the project
    # and it will automatically unzip and locate, so long as it follows formatting `tl_XXXX_us_county.zip` (i.e. `tl_2024_us_county.zip``)
    # https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/

    xlma_params = XLMAParams(
        dark_theme=True,
        color_unit='power_db',
        cartopy_paths= toolbox.append_county([])
    )

    if EXPORT_GENERAL_STATS:
        config_and_parser.export_general_stats(bucketed_strikes_indices, bucketed_lightning_correlations, events, config=lightning_configuration, xlma_params=xlma_params)

    if EXPORT_ALL_STRIKES:
        config_and_parser.export_all_strikes(bucketed_strikes_indices, events, config=lightning_configuration, xlma_params=xlma_params)

    if EXPORT_ALL_STRIKES_STITCHINGS:
        config_and_parser.export_strike_stitchings(bucketed_lightning_correlations, events, config=lightning_configuration, xlma_params=xlma_params)

    tprint("Finished generating plots")

if __name__ == '__main__':
    main()