import numpy as np
import pandas as pd
import pickle as pkl
import os
import hashlib
import re
import datetime
from tqdm import tqdm
from typing import List, Tuple, Optional
import multiprocessing
from collections import Counter
from . import toolbox
from .toolbox import tprint
from . import lightning_stitcher



# Global constants for cache handling.
RESULT_CACHE_FILE: str = "result_cache.pkl"

global_shutdown_event = None

def init_worker(shutdown_ev):
    global global_shutdown_event
    global_shutdown_event = shutdown_ev

def _group_process(args_list):
    """
    Process a subset of time groups to cluster lightning events into strikes.
    
    Parameters:
      args_list: Tuple containing:
        - all_x_values: NumPy array of x coordinates for all lightning events.
        - all_y_values: NumPy array of y coordinates for all lightning events.
        - all_z_values: NumPy array of z coordinates for all lightning events.
        - all_unix_values: NumPy array of Unix timestamps for all lightning events.
        - unique_groups: Array of unique group identifiers (time buckets) for the current chunk.
        - min_pts: Minimum number of events required to form a valid strike.
        - group_ids: Array mapping each event to its time group.
        - max_lightning_duration: Maximum allowed duration (in seconds) for a lightning strike.
        - max_dist_between_pts: Maximum allowed spatial distance (in meters) between events.
        - max_time_threshold: Maximum allowed time difference between consecutive events (seconds).
        - min_speed: Minimum allowed speed (in m/s) between events.
        - max_speed: Maximum allowed speed (in m/s) between events.
        
    Returns:
      List of lightning strike clusters, each represented as a list of event indices.
    """
    # Unpack input arguments.
    all_x_values, all_y_values, all_z_values, all_unix_values, unique_groups, min_pts, group_ids, max_lightning_duration, max_dist_between_pts, max_time_threshold, min_speed, max_speed = args_list

    lightning_strikes = []

    for group in unique_groups:

        if global_shutdown_event and global_shutdown_event.is_set():
            break

        group_indices = np.where(group_ids == group)[0]

        # Skip groups with fewer points than required.
        if len(group_indices) < min_pts:
            continue

        x_vals = all_x_values[group_indices]
        y_vals = all_y_values[group_indices]
        z_vals = all_z_values[group_indices]
        unix_vals = all_unix_values[group_indices]

        sub_groups = []  # Will hold potential lightning strikes for this group

        for j in range(len(x_vals)):
            if global_shutdown_event and global_shutdown_event.is_set():
                break
            event_x = x_vals[j]
            event_y = y_vals[j]
            event_z = z_vals[j]
            event_unix = unix_vals[j]

            # Finalize subgroups that have exceeded max_lightning_duration.
            lightning_strikes += [
                [group_indices[idx] for idx in sg["indices"]]
                for sg in sub_groups
                if event_unix - np.array(sg["unix"])[0] > max_lightning_duration and len(sg["indices"]) >= min_pts
            ]
            sub_groups = [
                sg for sg in sub_groups
                if event_unix - np.array(sg["unix"])[0] <= max_lightning_duration
            ]

            max_dist_squared = max_dist_between_pts ** 2
            min_speed_squared = min_speed ** 2
            max_speed_squared = max_speed ** 2

            found = False
            for sg in sub_groups:
                if global_shutdown_event and global_shutdown_event.is_set():
                    break
                # Convert list to array for vectorized operations.
                sg_unix = np.array(sg["unix"])
                dt_all = np.abs(event_unix - sg_unix)
                candidate_mask = dt_all <= max_time_threshold
                if not np.any(candidate_mask):
                    continue

                candidate_x = np.array(sg["x"])[candidate_mask]
                candidate_y = np.array(sg["y"])[candidate_mask]
                candidate_z = np.array(sg["z"])[candidate_mask]
                candidate_dt = dt_all[candidate_mask]

                dx = candidate_x - event_x
                dy = candidate_y - event_y
                dz = candidate_z - event_z
                distances_squared = dx * dx + dy * dy + dz * dz

                if np.any(distances_squared <= max_dist_squared):
                    # Check speed constraints.
                    candidate_dt_squared = candidate_dt * candidate_dt
                    dt_squared = np.where(candidate_dt_squared < 1e-5, 1e-5, candidate_dt_squared)
                    speeds_squared = distances_squared / dt_squared

                    if np.any((speeds_squared >= min_speed_squared) & (speeds_squared <= max_speed_squared)):
                        if event_unix - sg_unix[0] <= max_lightning_duration:
                            sg["indices"].append(j)
                            # Use list append rather than np.concatenate.
                            sg["x"].append(event_x)
                            sg["y"].append(event_y)
                            sg["z"].append(event_z)
                            sg["unix"].append(event_unix)
                            found = True
                            break

            if not found:
                # Start a new subgroup with list storage.
                sub_groups.append({
                    "indices": [j],
                    "x": [event_x],
                    "y": [event_y],
                    "z": [event_z],
                    "unix": [event_unix],
                })

        # Finalize any remaining valid subgroups.
        for sg in sub_groups:
            if len(sg["indices"]) >= min_pts:
                final_subgroup = np.array([group_indices[idx] for idx in sg["indices"]], dtype=np.uint32)
                lightning_strikes.append(final_subgroup)

    return lightning_strikes


NUM_CORES = 1
MAX_CHUNK_SIZE = 50000

def _bucket_dataframe_lightnings(
    df: pd.DataFrame,
    max_time_threshold: float,
    max_lightning_duration: float,
    max_dist_between_pts: float,
    max_speed: float,
    min_speed: float = 0,
    min_pts: int = 0,
) -> List[List[int]]:
    """
    Buckets the DataFrame into groups of lightning strikes based on temporal and spatial constraints.
    
    Steps:
      1. Sort events chronologically by 'time_unix'.
      2. Compute time differences between consecutive events.
      3. Group events into time buckets where the gap between consecutive events exceeds max_time_threshold.
      4. For each time bucket, further cluster events into potential lightning strikes based on:
         - A minimum number of events (min_pts).
         - Spatial proximity (max_dist_between_pts).
         - Speed constraints (min_speed and max_speed).
         - Maximum lightning duration (max_lightning_duration) to finalize clusters.
         
    Parameters:
      df (pd.DataFrame): DataFrame containing lightning event data.
      max_time_threshold (float): Maximum allowed time difference between consecutive events (seconds).
      max_lightning_duration (float): Maximum duration for a lightning strike (seconds).
      max_dist_between_pts (float): Maximum allowed spatial distance between events (meters).
      max_speed (float): Maximum allowed speed between events (m/s).
      min_speed (float, optional): Minimum allowed speed between events (m/s). Defaults to 0.
      min_pts (int, optional): Minimum number of events required for a valid lightning strike. Defaults to 0.
      
    Returns:
      List[List[int]]: A list of lightning strike clusters, each represented as a list of event indices.
    """
    df.sort_values(by="time_unix", inplace=True)
    time_unix_array = np.asarray(df["time_unix"].values)
    delta_t = np.diff(time_unix_array)

    # Group events by time threshold using cumulative sum.
    time_groups = np.concatenate(
        (
            np.array([0], dtype=np.int32),
            np.cumsum((delta_t > max_time_threshold).astype(np.int32)),
        )
    )
    tprint(time_groups)
    tprint("Processing the buckets.")

    group_ids = time_groups
    group_counter = Counter(group_ids)

    chunks = list(toolbox.chunk_items(group_counter, MAX_CHUNK_SIZE))

    all_x_values = df["x"].values
    all_y_values = df["y"].values
    all_z_values = df["z"].values
    all_unix_values = df["time_unix"].values

    shutdown_event = multiprocessing.Event()

    args_list = [
        (all_x_values, all_y_values, all_z_values, all_unix_values, chunk, min_pts, group_ids, max_lightning_duration, max_dist_between_pts, max_time_threshold, min_speed, max_speed)
        for chunk in chunks
    ]

    try:
        lightning_strikes: List[List[int]] = []
        if NUM_CORES > 1:
            with multiprocessing.Pool(processes=NUM_CORES, initializer=init_worker, initargs=(shutdown_event,)) as pool:
                for result in tqdm(pool.imap(_group_process, iterable=args_list), desc="Processing Chunks of Buckets",total=len(args_list)):
                    lightning_strikes += result
        else:
            for args in tqdm(args_list, desc="Processing Chunks of Buckets", total=len(args_list)):
                lightning_strikes += _group_process(args)
    except KeyboardInterrupt:
        shutdown_event.set()
        return None


    tprint("Passed groups:", len(lightning_strikes))
    return lightning_strikes


def _compute_cache_key(df: pd.DataFrame, params: dict) -> str:
    """
    Compute a unique cache key based on the DataFrame and bucketing parameters.
    
    The key is composed of:
      - DataFrame shape.
      - Minimum and maximum 'time_unix' values.
      - Sorted bucketing parameters.
      
    Parameters:
      df (pd.DataFrame): DataFrame containing lightning event data.
      params (dict): Bucketing parameters.
      
    Returns:
      str: MD5 hash representing the unique cache key.
    """
    key_str = f"{df.shape}_{df['time_unix'].min()}_{df['time_unix'].max()}_{sorted(params.items())}"
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


def delete_result_cache() -> None:
    """
    Delete the cached result file from the filesystem.
    """
    if os.path.exists(RESULT_CACHE_FILE):
        os.remove(RESULT_CACHE_FILE)


def _get_result_cache(
    df: pd.DataFrame, params: dict
) -> Optional[Tuple[List[List[int]], List[Tuple[int, int]], datetime.datetime]]:
    key = _compute_cache_key(df, params)
    max_cache_life_days = params.get("max_cache_life_days", 30)
    if os.path.exists(RESULT_CACHE_FILE):
        try:
            with open(RESULT_CACHE_FILE, "rb") as f:
                cache = pkl.load(f)
            if key in cache:
                filtered_groups, bucketed_correlations, time_saved = cache[key]
                now = datetime.datetime.now(tz=datetime.timezone.utc)
                if now - time_saved > datetime.timedelta(days=max_cache_life_days):
                    tprint("Cached result expired. Removing outdated cache entry.")
                    # Remove the expired cache entry and update the file.
                    del cache[key]
                    with open(RESULT_CACHE_FILE, "wb") as f:
                        pkl.dump(cache, f)
                    return None
                else:
                    tprint("Cache hit.")
                    return cache[key]
        except Exception as e:
            tprint(f"Cache load error: {e}")
    return None


def save_result_cache(
    df: pd.DataFrame, params: dict, result: Tuple[List[List[int]], List[Tuple[int, int]]]
) -> None:
    """
    Save the bucketing result in the cache with the computed key.
    
    Parameters:
      df (pd.DataFrame): DataFrame containing lightning event data.
      params (dict): Bucketing parameters.
      result (Tuple[List[List[int]], List[Tuple[int, int]]]): The bucketing result to be cached.
    """
    key = _compute_cache_key(df, params)
    cache = {}
    if os.path.exists(RESULT_CACHE_FILE):
        try:
            with open(RESULT_CACHE_FILE, "rb") as f:
                cache = pkl.load(f)
        except Exception as e:
            tprint(f"Cache load error: {e}")
            cache = {}
    cache[key] = result
    with open(RESULT_CACHE_FILE, "wb") as f:
        pkl.dump(cache, f)


def bucket_dataframe_lightnings(
    df: pd.DataFrame, params: dict
) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """
    Bucket lightning strikes in the dataframe using provided temporal and spatial parameters.
    
    Steps:
      1. Group events by time differences (using max_lightning_time_threshold).
      2. Cluster events within each time group based on spatial distance (max_lightning_dist) and speed constraints.
      3. Filter clusters that do not meet the minimum number of points (min_lightning_points).
      4. Cache the computed results for future reuse.
      
    Expected parameters in 'params':
      - max_lightning_dist (float): Maximum allowed spatial distance between events (meters).
      - max_lightning_speed (float): Maximum allowed speed between events (m/s).
      - min_lightning_speed (float): Minimum allowed speed between events (m/s).
      - min_lightning_points (int): Minimum number of events required for a valid lightning strike.
      - max_lightning_time_threshold (float): Maximum allowed time difference between consecutive events (seconds).
      - max_lightning_duration (float): Maximum duration for a lightning strike (seconds).
      
    Parameters:
      df (pd.DataFrame): DataFrame containing lightning event data.
      params (dict): Additional keyword arguments for bucketing behavior.
      
    Returns:
      Tuple[List[List[int]], List[Tuple[int, int]]]: A tuple containing:
         - A list of lightning strike clusters (each is a list of event indices).
         - A list of correlations between event indices.
    """
    use_cache = params.get("cache_results", False)

    if use_cache:
        cached_data = _get_result_cache(df, params)
        if cached_data is not None:
            filtered_groups, bucketed_correlations, time_saved = cached_data
            tprint("Using cached result from earlier")
            return filtered_groups, bucketed_correlations

    raw_groups = _bucket_dataframe_lightnings(
        df,
        max_time_threshold=params.get("max_lightning_time_threshold", 1),
        max_lightning_duration=params.get("max_lightning_duration", 20.0),
        max_dist_between_pts=params.get("max_lightning_dist", 50000),
        max_speed=params.get("max_lightning_speed", 299792.458),
        min_speed=params.get("min_lightning_speed", 0),
        min_pts=params.get("min_lightning_points", 300),
    )

    if raw_groups == None:
        return None, None

    bucketed_correlations = lightning_stitcher.stitch_lightning_strikes(raw_groups, df, params)

    filtered_groups: List[List[int]] = []

    # Gather unique indices from correlations.
    for correlations in bucketed_correlations:
        unique_indices_set = {idx for pair in correlations for idx in pair}
        unique_indices = list(unique_indices_set)

        filtered_groups.append(unique_indices)

    if use_cache:
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        save_result_cache(df, params, (filtered_groups, bucketed_correlations, now))

    return filtered_groups, bucketed_correlations


def export_as_csv(bucketed_strike_indices: List[List[int]], events: pd.DataFrame, output_dir: str) -> None:
    """
    Exports each lightning strike cluster to a CSV file in the specified output directory.
    
    Parameters:
      bucketed_strike_indices (List[List[int]]): List of clusters, where each cluster is a list of event indices.
      events (pd.DataFrame): DataFrame containing event data.
      output_dir (str): Directory where CSV files will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for indices in bucketed_strike_indices:
        # Extract events corresponding to the lightning strike and sort by time.
        strike_df = events.iloc[indices].sort_values(by="time_unix")
        start_time_unix = strike_df.iloc[0]["time_unix"]
        start_time_dt = datetime.datetime.fromtimestamp(
            start_time_unix, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")

        safe_start_time = re.sub(r'[<>:"/\\|?*]', '_', str(start_time_dt))
        output_filename = os.path.join(output_dir, f"{safe_start_time}.csv")
        counter = 1
        while os.path.exists(output_filename):
            output_filename = os.path.join(output_dir, f"{safe_start_time}_{counter}.csv")
            counter += 1

        strike_df.to_csv(output_filename, index=False)
        tprint(f"Exported lightning strike CSV to {output_filename}")
