import pandas as pd
import numpy as np
from typing import Tuple
from tqdm import tqdm


def filter_correlations_by_chain_size(correlations, min_pts, filter_point_to_self: bool = True):
    """
    Filter out correlations that do not belong to a connected chain with at least min_pts nodes.
    
    This function builds an undirected graph where each correlation (parent, child)
    represents an edge. It then uses depth-first search (DFS) to identify connected
    components. Only nodes within components that have at least min_pts nodes are considered valid.
    
    Parameters:
      correlations: List of tuples (parent, child) representing connections between events.
      min_pts: Minimum number of nodes required for a chain to be considered valid.
      
    Returns:
      A list of filtered correlations where both nodes belong to a valid chain.
    """

    # Build an undirected graph from the correlations.
    graph = {}
    for parent, child in correlations:
        graph.setdefault(parent, set()).add(child)
        graph.setdefault(child, set()).add(parent)
    
    visited = set()
    valid_nodes = set()
    
    # Use DFS to find connected components.
    for node in graph:
        if node not in visited:
            stack = [node]
            component = set()
            while stack:
                current = stack.pop()
                if current not in component:
                    component.add(current)
                    stack.extend(graph.get(current, []))
            visited |= component
            if len(component) >= min_pts:
                valid_nodes |= component
    
    # Filter correlations: both parent and child must be in a valid chain.
    return [(p, c) for (p, c) in correlations if p in valid_nodes and c in valid_nodes and (filter_point_to_self and p != c)]


def stitch_lightning_strike(strike_indeces: list[int], events: pd.DataFrame, params: dict) -> list[Tuple[(int, int)]]:
    """
    Build a chain of lightning strike nodes by connecting each strike to the closest preceding strike,
    subject to thresholds on time difference, spatial distance, and speed.
    
    Parameters:
      strike_indeces (list[int]): List of indices corresponding to lightning strike events.
      events (pd.DataFrame): DataFrame containing event data with columns such as "time_unix", "x", "y", and "z".
      params (dict): Additional filtering parameters including:
          - max_lightning_time_threshold (float): Maximum allowed time difference between consecutive points (default: 1 second).
          - max_lightning_dist (float): Maximum allowed distance between consecutive points (default: 50000 meters).
          - max_lightning_speed (float): Maximum allowed speed (default: 299792.458 m/s).
          - min_lightning_speed (float): Minimum allowed speed (default: 0 m/s).
          - min_lightning_points (int): Minimum number of points required for a valid lightning strike chain (default: 300).
    
    Returns:
      List of tuples (parent_index, child_index) representing valid correlations between lightning strike events.
    """
    # Retrieve filtering parameters.
    max_time_threshold = params.get("max_lightning_time_threshold", 1)
    max_dist_between_pts = params.get("max_lightning_dist", 50000)
    max_speed = params.get("max_lightning_speed", 299792.458)
    min_speed = params.get("min_lightning_speed", 0)
    min_pts = params.get("min_lightning_points", 300)


    # Sort the strike indices chronologically (using "time_unix").
    strike_indeces: list[int] = sorted(strike_indeces, key=lambda idx: events.loc[idx, "time_unix"])
    

    # Create a Series DataFrame for only the selected strikes.
    strike_series_df: pd.DataFrame = events.iloc[strike_indeces]

    # Cache the cupy arrays for the data columns.
    all_x = np.array(strike_series_df["x"].values)
    all_y = np.array(strike_series_df["y"].values)
    all_z = np.array(strike_series_df["z"].values)
    all_times = np.array(strike_series_df["time_unix"].values)

    # List to store nodes corresponding to each strike.
    parsed_indices: list[int] = []
    correlations: list[Tuple[(int, int)]] = []

    for i in range(len(strike_indeces)):
        current_indice = strike_indeces[i]

        if len(parsed_indices) > 0:
            # Get the current strike's coordinates and time.
            x1, y1, z1 = all_x[i], all_y[i], all_z[i]
            current_time = all_times[i]

            x_pre = all_x[:i]
            y_pre = all_y[:i]
            z_pre = all_z[:i]
            times_pre = all_times[:i]
            current_coords = np.array([x1, y1, z1])

            # Compute squared Euclidean distances.
            # We don't sqrt for optimization purposes.
            # We just do our math in squareds
            dx = x_pre - current_coords[0]
            dy = y_pre - current_coords[1]
            dz = z_pre - current_coords[2]
            distances_squared = dx * dx + dy * dy + dz * dz

            # Compute time differences (seconds).
            dt = current_time - times_pre

            dt_squared = (dt * dt)
            dt_squared = np.where(dt_squared < 1e-5, 1e-5, dt_squared)  # Avoid divide-by-zero

            # Compute squared speeds (m²/s²).
            speeds_squared = distances_squared / dt_squared

            # Precompute squared thresholds.
            max_dist_squared = max_dist_between_pts ** 2
            max_speed_squared = max_speed ** 2
            min_speed_squared = min_speed ** 2
            max_time_threshold_squared = max_time_threshold ** 2

            # Apply filtering mask using squared comparisons.
            mask = (distances_squared <= max_dist_squared)
            mask &= (speeds_squared <= max_speed_squared) 
            mask &= (speeds_squared >= min_speed_squared)
            mask &= (dt_squared <= max_time_threshold_squared)

            valid_indices = np.where(mask)[0]

            if valid_indices.size > 0:
                # Select the candidate with the minimum distance among those valid.
                valid_distances_squared = distances_squared[valid_indices]
                min_valid_idx = int(np.argmin(valid_distances_squared))
                candidate_idx = int(valid_indices[min_valid_idx])
                parent_indice = parsed_indices[candidate_idx]

                correlations.append((parent_indice, current_indice))

        # Save the current node.
        parsed_indices.append(current_indice)

    # Filter out correlations that are not connected to a lightning strike that contains min_pts pts
    correlations_filtered = filter_correlations_by_chain_size(correlations, min_pts, filter_point_to_self=True)

    return correlations_filtered


def stitch_lightning_strikes(bucketed_strike_indices: list[list[int]], events: pd.DataFrame, params: dict) -> list[list[Tuple[int, int]]]:
    """
    Process multiple groups of lightning strike indices and generate correlations for each group.
    
    This function applies the stitch_lightning_strike function on each group of strikes,
    and optionally combines groups with overlapping or intercepting times into a larger chain,
    based on temporal and spatial criteria.
    
    Parameters:
      bucketed_strike_indices (list[list[int]]): A list where each element is a list of strike event indices representing a group.
      events (pandas.DataFrame): DataFrame containing event data.
      params (dict): Additional parameters passed to stitch_lightning_strike and for combining groups, including:
          - combine_strikes_with_intercepting_times (bool): Whether to merge groups with intercepting time windows (default: True).
          - intercepting_times_extension_buffer (float): Extra time buffer (in seconds) added when checking intercepting groups (default: 10).
          - max_lightning_duration (float): Maximum allowed lightning duration for combining groups (default: 10).
          - intercepting_times_extension_max_distance (float): Maximum allowed distance (in meters) for intercepting groups (default: 15000).
    
    Returns:
      A list (one element per input group) where each element is a list of tuples (parent_index, child_index)
      representing correlations between lightning strike events.
    """
    # First, compute correlations for each strike group.
    bucketed_correlations = []
    for strike_indices in tqdm(bucketed_strike_indices, desc="Stitching Lightning Strikes", total=len(bucketed_strike_indices)):
        correlations = stitch_lightning_strike(strike_indices, events, params)
        bucketed_correlations.append(correlations)
    
    # Retrieve combining parameters.
    combine = params.get("combine_strikes_with_intercepting_times", True)
    ext_buffer = params.get("intercepting_times_extension_buffer", 10)
    max_duration = params.get("max_lightning_duration", 10)
    max_distance = params.get("intercepting_times_extension_max_distance", 15000)
    max_distance_sq = max_distance ** 2

    # Precompute event data arrays for fast lookup.
    all_times = events["time_unix"].values
    all_x = events["x"].values
    all_y = events["y"].values
    all_z = events["z"].values

    if combine:
        temp_bucketed_correlations = []
        for correlations in tqdm(bucketed_correlations, desc="Grouping Intercepting Lightning Strikes", total=len(bucketed_correlations)):
            if len(correlations) == 0:
                continue

            # Sort correlations by the parent's event time.
            sorted_corr = sorted(correlations, key=lambda corr: all_times[corr[0]])
            start_idx = sorted_corr[0][0]
            end_idx = sorted_corr[-1][-1]
            start_time = all_times[start_idx]
            end_time = all_times[end_idx]

            # Use the coordinates of the starting event for spatial comparisons.
            x1 = all_x[start_idx]
            y1 = all_y[start_idx]
            z1 = all_z[start_idx]
            
            result_found = False
            for i, temp_corr in enumerate(temp_bucketed_correlations):
                temp_start = all_times[temp_corr[0][0]]
                temp_end = all_times[temp_corr[-1][-1]] + ext_buffer

                # Check if the new group's time window is compatible with the existing group.
                within_duration = (max(abs(end_time - temp_start), abs(start_time - temp_end)) < max_duration)
                intercept_window = (temp_start < start_time < temp_end)

                if intercept_window and within_duration:
                    # Merge the groups: gather unique event indices.
                    unique_idx_set = {idx for pair in temp_corr for idx in pair}
                    unique_idx_list = list(unique_idx_set)
                    xs = all_x[unique_idx_list]
                    ys = all_y[unique_idx_list]
                    zs = all_z[unique_idx_list]

                    dx = xs - x1
                    dy = ys - y1
                    dz = zs - z1
                    dist_sq = dx * dx + dy * dy + dz * dz
                    if np.any(dist_sq <= max_distance_sq):
                        combined_corr = temp_corr + sorted_corr
                        unique_idx_set = {idx for pair in combined_corr for idx in pair}
                        unique_idx_list = list(unique_idx_set)

                        # re-stitch with new data
                        merged = stitch_lightning_strike(unique_idx_list, events, params)
                        merged = sorted(merged, key=lambda corr: all_times[corr[0]])
                        temp_bucketed_correlations[i] = merged
                        result_found = True
                        break
            
            if not result_found:
                temp_bucketed_correlations.append(sorted_corr)
        bucketed_correlations = temp_bucketed_correlations

        

    return bucketed_correlations


