"""
Lightning Data Stitching and Analysis Module

This module processes LYLOUT data files by:
  1. Parsing data files into an SQLite database.
  2. Extracting events into a Pandas DataFrame based on filters.
  3. Identifying lightning strikes from the events using multiprocessing.
  4. Exporting results as CSV files, plots, and animations.
"""

import os
import shutil
import numpy as np
import pandas as pd
from .number_crunchers import database_parser, lightning_bucketer, lightning_plotters, toolbox
from .number_crunchers.toolbox import tprint
from .number_crunchers.lightning_visualization import XLMAParams,\
    create_strike_gif, export_strike_gif, create_strike_image, export_strike_image, export_stats, export_bulk_to_folder
from typing import Tuple, List
from remote_functions import RemoteFunctions
from deprecation import deprecated
import datetime
import re
from tqdm import tqdm

rf = RemoteFunctions()

class LightningConfig:
    """
    Configuration settings for lightning data processing.
    """
    def __init__(self,
                 num_cores: int = 1,
                 lightning_data_folder: str = "lylout_files",
                 data_extension: str = ".dat",
                 cache_dir: str = "cache_dir",
                 csv_dir: str = "strikes_csv_files",
                 export_dir: str = "export",
                 strike_dir: str = "strikes",
                 strike_stitchings_dir: str = "strike_stitchings"):
        self.num_cores = num_cores
        self.lightning_data_folder = lightning_data_folder
        self.data_extension = data_extension

        self.cache_dir = cache_dir
        self.db_path = os.path.join(cache_dir, "lylout_db.db")
        self.cache_path = os.path.join(cache_dir, "os_cache.pkl")

        self.csv_dir = csv_dir
        self.export_dir = export_dir
        self.strike_dir = strike_dir
        self.strike_stitchings_dir = strike_stitchings_dir

        # Ensure required directories exist.
        os.makedirs(self.lightning_data_folder, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

server_sided_config_override: LightningConfig = None

@rf.as_remote_no_queue()
def get_lylout_files(config: LightningConfig) -> List[str]:
   """
    Retrieves a list of LYLOUT files from the designated lightning data folder.

    This function returns the filenames contained in the lightning data folder specified
    within the provided configuration. If a server-sited configuration override is active,
    that configuration is used instead of the provided one.

    Parameters:
      config (LightningConfig): The configuration object that holds the path to the lightning data folder.

    Returns:
      List[str]: A list of filenames found in the lightning data folder.
    """

   if server_sided_config_override:
       config = server_sided_config_override

   return os.listdir(config.lightning_data_folder) 

@rf.as_remote()
def remove_lylout_file(config: LightningConfig, filename: str) -> bool:
    """
    Removes a specified LYLOUT file from the lightning data folder.

    The function validates that the filename ends with the '.dat' extension and ensures that
    the file exists in the designated folder. If the file is not found, or if the filename does
    not meet the naming requirement, an appropriate exception is raised.

    Parameters:
      config (LightningConfig): The configuration object that holds the path to the lightning data folder.
      filename (str): The name of the LYLOUT file to be removed (must end with ".dat").

    Raises:
      NameError: If the filename does not end with ".dat".
      FileExistsError: If the specified file does not exist in the lightning data folder.
    """
    if server_sided_config_override:
       config = server_sided_config_override

    if not filename.endswith(".dat"):
        raise NameError("The LYLOUT file should end with .dat")
    
    filename = str(os.path.basename(filename)) # Remove the path
    
    full_path = os.path.join(config.lightning_data_folder, filename)

    if not os.path.exists(full_path):
        raise FileExistsError(f"File does not exist: {filename}")
    
    os.remove(full_path)
    return True

@rf.as_remote()
def upload_lylout_file(config: LightningConfig, filename: str, contents: str) -> bool:
    """
    Uploads a LYLOUT file by writing the provided contents into a file in the lightning data folder.

    The function validates that the filename ends with ".dat" and that the contents are predominantly
    text-like. If these validations pass, it writes the contents to a file in the designated folder.
    In the presence of a server-sited configuration override, that configuration is used.

    Parameters:
      config (LightningConfig): The configuration object that holds the path to the lightning data folder.
      filename (str): The name of the file to be uploaded (must end with ".dat").
      contents (str): The text content to be written into the file.

    Raises:
      NameError: If the filename does not end with ".dat".
      BufferError: If the contents are not considered to be mostly text.
    """
    if server_sided_config_override:
       config = server_sided_config_override

    if not filename.endswith(".dat"):
        raise NameError("The LYLOUT file should end with .dat")
    
    filename = str(os.path.basename(filename)) # Remove the path
    
    if not toolbox.is_mostly_text(contents):
        raise BufferError("The contents of the file must be mostly text.")
    
    full_path = os.path.join(config.lightning_data_folder, filename)

    with open(full_path, "w") as f:
        f.write(contents)

    return True
    
@rf.as_remote_no_queue()
def limit_to_n_points(bucketed_strikes_indices: list[list[int]],
                      bucketed_lightning_correlations: list[list[int, int]],
                      min_points_threshold: int):
    
    """
    Filters out buckets with fewer points than the specified threshold.

    Args:
        bucketed_strikes_indices: List of indices for each lightning strike.
        bucketed_lightning_correlations: List of correlated indices per strike.
        min_points_threshold: Minimum number of points required.

    Returns:
        tuple: Filtered (bucketed_strikes_indices, bucketed_lightning_correlations).
    """
    filtered_strikes = [lst for lst in bucketed_strikes_indices if len(lst) > min_points_threshold]
    filtered_correlations = [lst for lst in bucketed_lightning_correlations if len(lst) > min_points_threshold]
    return filtered_strikes, filtered_correlations

def _cache_and_parse(config: LightningConfig):
    """
    Retrieves LYLOUT files from the specified directory and caches the data into an SQLite database.
    Exits if no data files are found.
    
    Args:
        config: An instance of LightningConfig containing configuration settings.
    """
    if server_sided_config_override:
        config = server_sided_config_override

    files = os.listdir(config.lightning_data_folder)
    if not files:
        tprint(f"Please put lightning LYLOUT files in the directory '{config.lightning_data_folder}'")
        exit()

    # Parse and cache data into the SQLite database.
    database_parser.cache_and_parse_database(config.cache_dir,
                                               config.lightning_data_folder,
                                               config.data_extension,
                                               config.db_path,
                                               config.cache_path)
    # Display available headers from the database.
    tprint("Headers:", database_parser.get_headers(config.db_path))

@rf.as_remote()
def cache_and_parse(config: LightningConfig):
    """
    Retrieves LYLOUT files from the specified directory and caches the data into an SQLite database.
    Exits if no data files are found.
    
    Args:
        config: An instance of LightningConfig containing configuration settings.
    """
    if server_sided_config_override:
        config = server_sided_config_override
    return _cache_and_parse(config)

@rf.as_remote()
def get_headers(config: LightningConfig) -> List[str]:
    """
    Returns a list of headers from the database
    """
    if server_sided_config_override:
        config = server_sided_config_override
        
    _cache_and_parse(config) # Cache and parse
    return database_parser.get_headers(config.db_path)

@rf.as_remote()
def get_events(filters, config: LightningConfig) -> pd.DataFrame:
    """
    Retrieves event data from the SQLite database based on the provided filters.

    Args:
        filters: Filter criteria for the query.
        config: An instance of LightningConfig.

    Returns:
        pd.DataFrame: DataFrame containing event data.
    """
    if server_sided_config_override:
        config = server_sided_config_override

    tprint("Obtaining datapoints from database. This may take some time...")
    _cache_and_parse(config) # Cache and parse

    events = database_parser.query_events_as_dataframe(filters, config.db_path)
    if events.empty:
        tprint("Data too restrained")
    return events

@rf.as_remote()
def get_events_and_bucket_dataframe_lightnings(filters, config: LightningConfig, params) -> tuple[pd.DataFrame, List[List[int]], List[Tuple[int, int]]]:
    """
    Retrieves event data and buckets lightning strikes from the events.

    This function queries the event data using the provided filters and then
    calls an internal routine to bucket the events into lightning strikes based
    on the specified parameters. If no events are found, bucketing is skipped.

    Parameters:
      filters: Criteria to filter the event data.
      config (LightningConfig): Configuration with paths and settings.
      params: Parameters for the lightning bucketing process.

    Returns:
      tuple:
        - pd.DataFrame: DataFrame of event data.
        - List[List[int]]: Buckets of indices representing lightning strikes.
        - List[Tuple[int, int]]: Buckets of correlated indices for strikes.
    """
    if server_sided_config_override:
        config = server_sided_config_override

    _cache_and_parse(config) # Cache and parse    

    events = database_parser.query_events_as_dataframe(filters, config.db_path)
    
    bucketed_strikes_indices, bucketed_lightning_correlations = None, None
    if events.empty:
        tprint("Data too restrained")
    else:
        bucketed_strikes_indices, bucketed_lightning_correlations = _bucket_dataframe_lightnings(events=events, config=config, params=params)

    return events, bucketed_strikes_indices, bucketed_lightning_correlations


def _bucket_dataframe_lightnings(events, config: LightningConfig, params):
    # Enable caching for the bucketer.
    lightning_bucketer.RESULT_CACHE_FILE = os.path.join(config.cache_dir, "result_cache.pkl")

    # Set processing parameters.
    lightning_bucketer.NUM_CORES = config.num_cores
    lightning_bucketer.MAX_CHUNK_SIZE = 50000

    bucketed_strikes_indices, bucketed_lightning_correlations = lightning_bucketer.bucket_dataframe_lightnings(events, params)
    if not bucketed_strikes_indices:
        tprint("Data too restrained.")
        exit()
    tprint("Created buckets of nodes that resemble a lightning strike")
    return bucketed_strikes_indices, bucketed_lightning_correlations

@rf.as_remote()
def bucket_dataframe_lightnings(events: pd.DataFrame, config: LightningConfig, params) -> tuple[List[List[int]], List[Tuple[int, int]]]:
    """
    Buckets events into lightning strikes based on provided parameters, using caching and multiprocessing.

    Args:
        events: DataFrame containing event data.
        config: An instance of LightningConfig.
        params: Parameters for bucketing lightning strikes.

    Returns:
        tuple: (bucketed_strikes_indices, bucketed_lightning_correlations)
    """
    if server_sided_config_override:
        config = server_sided_config_override

    return _bucket_dataframe_lightnings(events=events, config=config, params=params)
    

def display_stats(events: pd.DataFrame, bucketed_strikes_indices: list[list[int]]):
    """
    Computes and displays statistics based on the lightning strike buckets.

    Args:
        events: DataFrame containing event data.
        bucketed_strikes_indices: Buckets of indices for lightning strikes.
    """
    total_points_passed = 0
    strike_durations = []

    for strike in bucketed_strikes_indices:
        start_time_unix = events.iloc[strike[0]]["time_unix"]
        end_time_unix = events.iloc[strike[-1]]["time_unix"]
        total_points_passed += len(strike)
        strike_durations.append(end_time_unix - start_time_unix)

    total_pts = len(events)
    pct = (total_points_passed / total_pts) * 100
    tprint(f"Passed points: {total_points_passed} out of {total_pts} points ({pct:.2f}%)")
    avg_time = np.average(strike_durations)
    tprint(f"Average lightning strike time: {avg_time:.2f} seconds")
    avg_bucket_size = int(total_pts / len(bucketed_strikes_indices))
    tprint(f"Average bucket size: {avg_bucket_size} points")
    tprint(f"Number of buckets: {len(bucketed_strikes_indices)}")

@rf.as_remote()
def delete_sql_database(config: LightningConfig):
    """
    This function deletes the entire sql database (Excluding LYLOUT files)
    This includes the pickled cache
    """
    if server_sided_config_override:
        config = server_sided_config_override

    shutil.rmtree(config.cache_dir)

@rf.as_remote()
def delete_pkl_cache(config: LightningConfig):
    """
    This function deletes the pickled cache
    """
    if server_sided_config_override:
        config = server_sided_config_override

    lightning_bucketer.RESULT_CACHE_FILE = os.path.join(config.cache_dir, "result_cache.pkl")
    lightning_bucketer.delete_result_cache()

def export_as_csv(bucketed_strikes_indices: list[list[int]], events: pd.DataFrame, config: LightningConfig):
    """
    Exports the lightning strikes data as CSV files.

    Args:
        bucketed_strikes_indices: Buckets of indices for lightning strikes.
        events: DataFrame containing event data.
        config: An instance of LightningConfig.
    """
    if server_sided_config_override:
        config = server_sided_config_override

    tprint("Exporting CSV data")
    if os.path.exists(config.csv_dir):
        shutil.rmtree(config.csv_dir)
    os.makedirs(config.csv_dir, exist_ok=True)
    lightning_bucketer.export_as_csv(bucketed_strikes_indices, events, output_dir=config.csv_dir)
    tprint("Finished exporting as CSV")

def export_general_stats(bucketed_strikes_indices: list[list[int]],
                         bucketed_lightning_correlations: list[list[int, int]],
                         events: pd.DataFrame,
                         config: LightningConfig,
                         xlma_params: XLMAParams = XLMAParams(),
                         _include_deprecated:bool = False):
    """
    Exports various plots and statistics for the lightning strikes.

    Args:
        bucketed_strikes_indices: Buckets of indices for lightning strikes.
        bucketed_lightning_correlations: Buckets of correlated indices.
        events: DataFrame containing event data.
        config: An instance of LightningConfig.
    """
    if server_sided_config_override:
        config = server_sided_config_override

    if os.path.exists(config.export_dir):
        shutil.rmtree(config.export_dir)
    os.makedirs(config.export_dir, exist_ok=True)

    tprint("Plotting stats")
    export_path = os.path.join(config.export_dir, "strike_pts_over_time")
    export_stats(xlma_params=xlma_params, events=events, bucketed_indeces=bucketed_strikes_indices, export_path=export_path + ".tiff")
    if _include_deprecated:
        lightning_plotters.plot_strikes_over_time(bucketed_strikes_indices, events, output_filename=export_path + ".png")

    largest_strike = max(bucketed_strikes_indices, key=len)
    largest_stitch = max(bucketed_lightning_correlations, key=len)

    tprint("Exporting XLMA diagram of largest Instance")
    strike_image, _ = create_strike_image(xlma_params, events, largest_strike, largest_stitch)
    export_path = os.path.join(config.export_dir, "most_pts_xlma.tiff")
    export_strike_image(strike_image, export_path)

    strike_gif, _ = create_strike_gif(xlma_params, events, largest_strike, largest_stitch)
    export_path = os.path.join(config.export_dir, "most_pts_xlma.gif")
    export_strike_gif(strike_gif, export_path)


    if _include_deprecated:
        tprint("Exporting largest instance")
        export_path = os.path.join(config.export_dir, "most_pts")
        lightning_plotters.plot_avg_power_map(largest_strike, events, output_filename=export_path + ".png", transparency_threshold=-1)
        lightning_plotters.generate_strike_gif(largest_strike, events, output_filename=export_path + ".gif", transparency_threshold=-1)

        tprint("Exporting largest stitched instance")
        export_path = os.path.join(config.export_dir, "most_pts_stitched")
        lightning_plotters.plot_lightning_stitch(largest_stitch, events, export_path + ".png")
        lightning_plotters.plot_lightning_stitch_gif(largest_stitch, events, output_filename=export_path + ".gif")

        tprint("Exporting all strike points")
        export_path = os.path.join(config.export_dir, "all_pts")
        combined_strikes = [idx for strike in bucketed_strikes_indices for idx in strike]
        lightning_plotters.plot_avg_power_map(combined_strikes, events, output_filename=export_path + ".png", transparency_threshold=-1)
        lightning_plotters.generate_strike_gif(combined_strikes, events, output_filename=export_path + ".gif", transparency_threshold=-1)

def export_all_strikes(bucketed_strikes_indices: list[list[int]], 
                       events: pd.DataFrame, 
                       config: LightningConfig,
                       xlma_params: XLMAParams = XLMAParams(),
                       _include_deprecated:bool = False):
    """
    Exports heatmap plots for all lightning strikes.

    Args:
        bucketed_strikes_indices: Buckets of indices for lightning strikes.
        events: DataFrame containing event data.
        config: An instance of LightningConfig.
    """
    if server_sided_config_override:
        config = server_sided_config_override

    if os.path.exists(config.strike_dir):
        shutil.rmtree(config.strike_dir)
    os.makedirs(config.strike_dir, exist_ok=True)


    export_bulk_to_folder(events=events, output_dir=config.strike_dir, bucketed_strike_indices=bucketed_strikes_indices, bucketed_strike_correlations=None, xlma_params=xlma_params)

            
    if _include_deprecated:
        tprint("Plotting all strikes as a heatmap")
        lightning_plotters.plot_all_strikes(bucketed_strikes_indices, events, config.strike_dir, config.num_cores,
                                            sigma=1.5, transparency_threshold=-1)
        lightning_plotters.plot_all_strikes(bucketed_strikes_indices, events, config.strike_dir, config.num_cores,
                                            as_gif=True, sigma=1.5, transparency_threshold=-1)
        tprint("Finished plotting strikes as a heatmap")

@deprecated(details="Plotly uses Kaleidoscope. Currently, Kaleidoscope engine has issues saving images. This function will continue to exist but will no longer be supported.")
def export_strike_stitchings(bucketed_lightning_correlations: list[list[int, int]], 
                             events: pd.DataFrame, 
                             config: LightningConfig,
                             xlma_params: XLMAParams = XLMAParams(),
                             _include_deprecated:bool = False):
    """
    Exports plots and animations for stitched lightning strikes.

    Args:
        bucketed_lightning_correlations: Buckets of correlated indices.
        events: DataFrame containing event data.
        config: An instance of LightningConfig.
    """
    if server_sided_config_override:
        config = server_sided_config_override

    tprint("Plotting all strike stitchings")
    if os.path.exists(config.strike_stitchings_dir):
        shutil.rmtree(config.strike_stitchings_dir)
    os.makedirs(config.strike_stitchings_dir, exist_ok=True)

    bucketed_strikes_indices = []
    for strike_stitchings in bucketed_lightning_correlations:
        strike_indices = set()
        for (parent_idx, child_idx) in strike_stitchings:
            strike_indices.add(parent_idx)
            strike_indices.add(child_idx)
        bucketed_strikes_indices.append(list(strike_indices))
        
    export_bulk_to_folder(events=events, output_dir=config.strike_stitchings_dir, bucketed_strike_indices=bucketed_strikes_indices, bucketed_strike_correlations=bucketed_lightning_correlations, xlma_params=xlma_params, num_cores=config.num_cores)


    if _include_deprecated:
        lightning_plotters.plot_all_strike_stitchings(bucketed_lightning_correlations, events, config.strike_stitchings_dir, config.num_cores)
        lightning_plotters.plot_all_strike_stitchings(bucketed_lightning_correlations, events, config.strike_stitchings_dir, config.num_cores,
                                                   as_gif=True)
    tprint("Finished outputting stitchings")
