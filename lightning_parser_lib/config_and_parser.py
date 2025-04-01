"""
Lightning Data Stitching and Analysis Module

This module processes LYLOUT data files (e.g. "LYLOUT_20220712_pol.exported.dat") by:
  1. Parsing data files into an SQLite database.
  2. Extracting events into a Pandas DataFrame based on user-specified filters.
  3. Identifying lightning strikes from the events using multiprocessing.
  4. Exporting results as CSV files, plots, and animations.

Usage:
  Ensure that your LYLOUT data files are placed in the "lylout_files" directory.
  Adjust the configuration settings below as necessary.
"""

import os
import shutil
import numpy as np
import pandas as pd
from .number_crunchers import database_parser as database_parser
from .number_crunchers import lightning_bucketer as lightning_bucketer
from .number_crunchers import lightning_plotters as lightning_plotters
from .number_crunchers.toolbox import tprint

####################################################################################
# Configuration and User-specified Settings
####################################################################################

# CPU utilization settings for multiprocessing.
NUM_CORES = 1

# Data folder and file extension for LYLOUT files.
LIGHTNING_DATA_FOLDER = "lylout_files"  # Directory for LYLOUT files.
DATA_EXTENSION = ".dat"
os.makedirs(LIGHTNING_DATA_FOLDER, exist_ok=True)

# Cache settings: directories and file paths.
CACHE_DIR = "cache_dir"  # Directory for cache and database files.
os.makedirs(CACHE_DIR, exist_ok=True)
DB_PATH = os.path.join(CACHE_DIR, "lylout_db.db")  # SQLite database file.
CACHE_PATH = os.path.join(CACHE_DIR, "os_cache.pkl")  # Pickle file for caching.

# Export options.
CSV_DIR = "strikes_csv_files"  # Directory to hold CSV files.
EXPORT_DIR = "export"       # Directory for general exported charts/plots.
STRIKE_DIR = "strikes"      # Directory for all strikes plots.
STRIKE_STITCHINGS_DIR = "strike_stitchings"  # Directory for stitched strikes plots.

####################################################################################
# Helper Functions
####################################################################################

def limit_to_n_points(bucketed_strikes_indices: list[list[int]], bucketed_lightning_correlations: list[list[int, int]], min_points_threshold):
    """
    Filters out buckets with fewer points than the specified threshold.

    Args:
        bucketed_strikes_indices (list[list[int]]): List of indices for each lightning strike.
        bucketed_lightning_correlations (list[list[int, int]]): List of correlated indices per strike.
        min_points_threshold (int): Minimum number of points required.

    Returns:
        tuple: Filtered (bucketed_strikes_indices, bucketed_lightning_correlations).
    """
    filtered_strikes = [lst for lst in bucketed_strikes_indices if len(lst) > min_points_threshold]
    filtered_correlations = [lst for lst in bucketed_lightning_correlations if len(lst) > min_points_threshold]
    return filtered_strikes, filtered_correlations


def cache_and_parse():
    """
    Retrieves LYLOUT files from the specified directory, caches the data into an SQLite database,
    and prints out the available database headers.
    
    Exits if no data files are found.
    """
    files = os.listdir(LIGHTNING_DATA_FOLDER)
    if len(files) == 0:
        tprint(f"Please put lightning LYLOUT files in the directory '{LIGHTNING_DATA_FOLDER}'")
        exit()

    # Parse and cache data into the SQLite database.
    database_parser.cache_and_parse_database(CACHE_DIR, LIGHTNING_DATA_FOLDER, DATA_EXTENSION, DB_PATH, CACHE_PATH)

    # Display available headers from the database.
    tprint("Headers:", database_parser.get_headers(DB_PATH))


def get_events(filters) -> pd.DataFrame:
    """
    Retrieves event data from the SQLite database based on the provided filters.

    Args:
        filters (list[tuple]): Filter criteria for the query.

    Returns:
        pd.DataFrame: DataFrame containing event data.
    """
    tprint("Obtaining datapoints from database. This may take some time...")
    events = database_parser.query_events_as_dataframe(filters, DB_PATH)
    if events.empty:
        tprint("Data too restrained")
    return events


def bucket_dataframe_lightnings(events: pd.DataFrame, **params):
    """
    Buckets events into lightning strikes based on provided parameters. Utilizes caching and multiprocessing.

    Args:
        events (pd.DataFrame): DataFrame containing event data.
        **params: Parameters for bucketing lightning strikes.

    Returns:
        tuple: A tuple containing:
            - bucketed_strikes_indices (list[list[int]]): Buckets of indices for lightning strikes.
            - bucketed_lightning_correlations (list[list[int, int]]): Buckets of correlated indices.
    """
    # Enable caching for the bucketer.
    lightning_bucketer.USE_CACHE = True
    lightning_bucketer.RESULT_CACHE_FILE = os.path.join(CACHE_DIR, "result_cache.pkl")

    # Set processing parameters.
    lightning_bucketer.NUM_CORES = NUM_CORES
    lightning_bucketer.MAX_CHUNK_SIZE = 50000

    # Bucket events into lightning strikes.
    bucketed_strikes_indices, bucketed_lightning_correlations = lightning_bucketer.bucket_dataframe_lightnings(events, **params)
    if len(bucketed_strikes_indices) == 0:
        tprint("Data too restrained.")
        exit()
    tprint("Created buckets of nodes that resemble a lightning strike")
    return bucketed_strikes_indices, bucketed_lightning_correlations


def display_stats(events: pd.DataFrame, bucketed_strikes_indices: list[list[int]]):
    """
    Computes and displays statistics based on the lightning strike buckets.

    Args:
        events (pd.DataFrame): DataFrame containing event data.
        bucketed_strikes_indices (list[list[int]]): Buckets of indices for lightning strikes.
    """
    total_points_passed = 0
    strike_durations = []

    # Iterate through each strike bucket to calculate statistics.
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


def export_as_csv(bucketed_strikes_indices: list[list[int]], events: pd.DataFrame):
    """
    Exports the lightning strikes data as CSV files.

    Args:
        bucketed_strikes_indices (list[list[int]]): Buckets of indices for lightning strikes.
        events (pd.DataFrame): DataFrame containing event data.
    """
    tprint("Exporting CSV data")
    if os.path.exists(CSV_DIR):
        shutil.rmtree(CSV_DIR)
    os.makedirs(CSV_DIR, exist_ok=True)
    lightning_bucketer.export_as_csv(bucketed_strikes_indices, events, output_dir=CSV_DIR)
    tprint("Finished exporting as CSV")


def export_general_stats(bucketed_strikes_indices: list[list[int]], bucketed_lightning_correlations: list[list[int, int]], events: pd.DataFrame):
    """
    Exports various plots and statistics:
      - Strike points over time.
      - Largest strike instance and its animation.
      - Largest stitched instance and its animation.
      - Combined strike overview.

    Args:
        bucketed_strikes_indices (list[list[int]]): Buckets of indices for lightning strikes.
        bucketed_lightning_correlations (list[list[int, int]]): Buckets of correlated indices.
        events (pd.DataFrame): DataFrame containing event data.
    """
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # Plot strikes over time.
    tprint("Plotting strike points over time")
    export_path = os.path.join(EXPORT_DIR, "strike_pts_over_time")
    lightning_plotters.plot_strikes_over_time(bucketed_strikes_indices, events, output_filename=export_path + ".png")

    # Export the largest strike instance.
    tprint("Exporting largest instance")
    export_path = os.path.join(EXPORT_DIR, "most_pts")
    largest_strike = max(bucketed_strikes_indices, key=len)
    lightning_plotters.plot_avg_power_map(largest_strike, events, output_filename=export_path + ".png", transparency_threshold=-1)
    lightning_plotters.generate_strike_gif(largest_strike, events, output_filename=export_path + ".gif", transparency_threshold=-1)

    # Export the largest stitched instance.
    tprint("Exporting largest stitched instance")
    export_path = os.path.join(EXPORT_DIR, "most_pts_stitched")
    largest_stitch = max(bucketed_lightning_correlations, key=len)
    lightning_plotters.plot_lightning_stitch(largest_stitch, events, export_path + ".png")
    lightning_plotters.plot_lightning_stitch_gif(largest_stitch, events, output_filename=export_path + ".gif")

    # Export combined view of all strikes.
    tprint("Exporting all strikes")
    export_path = os.path.join(EXPORT_DIR, "all_pts")
    combined_strikes = [idx for strike in bucketed_strikes_indices for idx in strike]
    lightning_plotters.plot_avg_power_map(combined_strikes, events, output_filename=export_path + ".png", transparency_threshold=-1)
    lightning_plotters.generate_strike_gif(combined_strikes, events, output_filename=export_path + ".gif", transparency_threshold=-1)

    tprint("Number of points within timeframe:", len(combined_strikes))


def export_all_strikes(bucketed_strikes_indices: list[list[int]], events: pd.DataFrame):
    """
    Exports heatmap plots for all lightning strikes.

    Args:
        bucketed_strikes_indices (list[list[int]]): Buckets of indices for lightning strikes.
        events (pd.DataFrame): DataFrame containing event data.
    """
    if os.path.exists(STRIKE_DIR):
        shutil.rmtree(STRIKE_DIR)
    os.makedirs(STRIKE_DIR, exist_ok=True)

    tprint("Plotting all strikes as a heatmap")
    lightning_plotters.plot_all_strikes(bucketed_strikes_indices, events, STRIKE_DIR, NUM_CORES,
                                         sigma=1.5, transparency_threshold=-1)
    lightning_plotters.plot_all_strikes(bucketed_strikes_indices, events, STRIKE_DIR, NUM_CORES,
                                         as_gif=True, sigma=1.5, transparency_threshold=-1)
    tprint("Finished plotting strikes as a heatmap")


def export_strike_stitchings(bucketed_lightning_correlations: list[list[int, int]], events: pd.DataFrame):
    """
    Exports plots and animations for stitched lightning strikes.

    Args:
        bucketed_lightning_correlations (list[list[int, int]]): Buckets of correlated indices.
        events (pd.DataFrame): DataFrame containing event data.
    """
    tprint("Plotting all strike stitchings")
    if os.path.exists(STRIKE_STITCHINGS_DIR):
        shutil.rmtree(STRIKE_STITCHINGS_DIR)
    lightning_plotters.plot_all_strike_stitchings(bucketed_lightning_correlations, events, STRIKE_STITCHINGS_DIR, NUM_CORES)
    lightning_plotters.plot_all_strike_stitchings(bucketed_lightning_correlations, events, STRIKE_STITCHINGS_DIR, NUM_CORES,
                                                   as_gif=True)
    tprint("Finished outputting stitchings")
