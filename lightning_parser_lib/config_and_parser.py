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
from .number_crunchers import database_parser, lightning_bucketer, lightning_plotters
from .number_crunchers.toolbox import tprint
from typing import Tuple, List
from remote_functions import RemoteFunctions

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

@rf.as_remote()
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

@rf.as_remote()
def get_headers(config: LightningConfig) -> List[str]:
    """
    Returns a list of headers from the database
    """
    if server_sided_config_override:
        config = server_sided_config_override
    return database_parser.get_headers(config.db_path)

@rf.as_remote()
def cache_and_parse(config: LightningConfig):
    """
    Retrieves LYLOUT files from the specified directory and caches the data into an SQLite database.
    Exits if no data files are found.
    
    Args:
        config: An instance of LightningConfig containing configuration settings.
    """
    if server_sided_config_settings:
        config = server_sided_config_settings

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
    events = database_parser.query_events_as_dataframe(filters, config.db_path)
    if events.empty:
        tprint("Data too restrained")
    return events

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

@rf.as_remote()
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

@rf.as_remote()
def export_general_stats(bucketed_strikes_indices: list[list[int]],
                         bucketed_lightning_correlations: list[list[int, int]],
                         events: pd.DataFrame,
                         config: LightningConfig):
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

    os.makedirs(config.export_dir, exist_ok=True)

    tprint("Plotting strike points over time")
    export_path = os.path.join(config.export_dir, "strike_pts_over_time")
    lightning_plotters.plot_strikes_over_time(bucketed_strikes_indices, events, output_filename=export_path + ".png")

    tprint("Exporting largest instance")
    export_path = os.path.join(config.export_dir, "most_pts")
    largest_strike = max(bucketed_strikes_indices, key=len)
    lightning_plotters.plot_avg_power_map(largest_strike, events, output_filename=export_path + ".png", transparency_threshold=-1)
    lightning_plotters.generate_strike_gif(largest_strike, events, output_filename=export_path + ".gif", transparency_threshold=-1)

    tprint("Exporting largest stitched instance")
    export_path = os.path.join(config.export_dir, "most_pts_stitched")
    largest_stitch = max(bucketed_lightning_correlations, key=len)
    lightning_plotters.plot_lightning_stitch(largest_stitch, events, export_path + ".png")
    lightning_plotters.plot_lightning_stitch_gif(largest_stitch, events, output_filename=export_path + ".gif")

    tprint("Exporting all strikes")
    export_path = os.path.join(config.export_dir, "all_pts")
    combined_strikes = [idx for strike in bucketed_strikes_indices for idx in strike]
    lightning_plotters.plot_avg_power_map(combined_strikes, events, output_filename=export_path + ".png", transparency_threshold=-1)
    lightning_plotters.generate_strike_gif(combined_strikes, events, output_filename=export_path + ".gif", transparency_threshold=-1)

    tprint("Number of points within timeframe:", len(combined_strikes))

@rf.as_remote()
def export_all_strikes(bucketed_strikes_indices: list[list[int]], events: pd.DataFrame, config: LightningConfig):
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

    tprint("Plotting all strikes as a heatmap")
    lightning_plotters.plot_all_strikes(bucketed_strikes_indices, events, config.strike_dir, config.num_cores,
                                         sigma=1.5, transparency_threshold=-1)
    lightning_plotters.plot_all_strikes(bucketed_strikes_indices, events, config.strike_dir, config.num_cores,
                                         as_gif=True, sigma=1.5, transparency_threshold=-1)
    tprint("Finished plotting strikes as a heatmap")

@rf.as_remote()
def export_strike_stitchings(bucketed_lightning_correlations: list[list[int, int]], events: pd.DataFrame, config: LightningConfig):
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
    lightning_plotters.plot_all_strike_stitchings(bucketed_lightning_correlations, events, config.strike_stitchings_dir, config.num_cores)
    lightning_plotters.plot_all_strike_stitchings(bucketed_lightning_correlations, events, config.strike_stitchings_dir, config.num_cores,
                                                   as_gif=True)
    tprint("Finished outputting stitchings")
