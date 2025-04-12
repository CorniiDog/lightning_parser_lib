import copy
import math
import numpy as np
from scipy.stats import skew, kurtosis
from typing import List, Tuple
import pandas as pd


def generate_prestats(events: pd.DataFrame,
                      bucketed_strikes_indices: List[List[int]], 
                      bucketed_lightning_correlations: List[List[Tuple[int, int]]]):
    """
    Generate pre-statistics for lightning events using event features.

    This function computes distances and speeds between correlated pairs of lightning events 
    and aggregates additional event statistics ('reduced_chi2', 'power', and 'power_db') for each bucket.
    The overall statistics are returned both as an aggregated dictionary and as a list of dictionaries for each bucket.

    Parameters:
      events (pd.DataFrame): The DataFrame containing event data with required columns: 'x', 'y', 'z', 
                             'time_unix', 'reduced_chi2', 'power', and 'power_db'.
      bucketed_strikes_indices (List[List[int]]): A list of lists where each sublist contains indices of events belonging to a specific bucket.
      bucketed_lightning_correlations (List[List[Tuple[int, int]]]): A list of lists where each sublist contains tuple pairs of indices 
                                                                     corresponding to correlated lightning events used to compute distances and speeds.

    Returns:
      Tuple[dict, List[dict]]: A tuple with two elements:
        - overall_detailed_stats (dict): A dictionary containing aggregated statistics ('distance', 'speed', 'reduced_chi2', 'power', 'power_db') across all buckets.
        - bucketed_stats (List[dict]): A list of dictionaries, each containing detailed statistics for a specific bucket.
    """
    
    stats_template = {
      "distance": [],
      "speed": [],
      "reduced_chi2": [],
      "power": [],
      "power_db": []
    }

    overall_detailed_stats = copy.deepcopy(stats_template)
    bucketed_stats = []
    for i, strikes_indices in enumerate(bucketed_strikes_indices):
        detailed_stats = copy.deepcopy(stats_template)

        strike_correlations = bucketed_lightning_correlations[i]

        for correlation in strike_correlations:
            correlation_sub_df = events.iloc[[correlation[0], correlation[1]]]

            x_values = correlation_sub_df['x'].values
            y_values = correlation_sub_df['y'].values
            z_values = correlation_sub_df['z'].values
            time_unix_values = correlation_sub_df['time_unix'].values
            
            distance = math.sqrt((x_values[1]-x_values[0])**2 + (y_values[1]-y_values[0])**2 + (z_values[1]-z_values[0]) ** 2)
            dt = abs(time_unix_values[1] - time_unix_values[0])

            if dt == 0:
                dt = 1e-10

            speed = distance/dt

            detailed_stats["speed"].append(speed)
            detailed_stats["distance"].append(distance)
            overall_detailed_stats["speed"].append(speed)
            overall_detailed_stats["distance"].append(distance)

        events_in_bucket = events.iloc[strikes_indices]

        detailed_stats["reduced_chi2"] += events_in_bucket["reduced_chi2"].values.tolist()
        detailed_stats["power"] += events_in_bucket["power"].values.tolist()
        detailed_stats["power_db"] += events_in_bucket["power_db"].values.tolist()
        overall_detailed_stats["reduced_chi2"] += events_in_bucket["reduced_chi2"].values.tolist()
        overall_detailed_stats["power"] += events_in_bucket["power"].values.tolist()
        overall_detailed_stats["power_db"] += events_in_bucket["power_db"].values.tolist()

        bucketed_stats.append(detailed_stats)

    return overall_detailed_stats, bucketed_stats


def compute_detailed_stats(overall_detailed_stats: dict) -> dict:
    """
    Compute statistical summaries for each aggregated event statistic.

    This function calculates descriptive statistics including the minimum, first quartile (q1), median, third quartile (q3),
    average, maximum, standard deviation, interquartile range (IQR), skewness, and excess kurtosis for each key in the provided 
    overall_detailed_stats dictionary.

    Parameters:
      overall_detailed_stats (dict): A dictionary where keys are statistic names ('distance', 'speed', 'reduced_chi2', 
                                     'power', 'power_db') and values are lists of corresponding numerical values.

    Returns:
      dict: A dictionary mapping each statistic name to another dictionary containing the computed summaries with keys 
            'min', 'q1', 'median', 'q3', 'average', 'max', 'std', 'IQR', 'skewness', and 'kurtosis' (excess kurtosis).
    """
    result = {}

    for key, stat_values in overall_detailed_stats.items():
        q1 = np.percentile(stat_values, 25)
        q2 = np.median(stat_values)
        q3 = np.percentile(stat_values, 75)
        iqr = q3 - q1

        result[key] = {
            "min": np.min(stat_values),
            "q1": q1,
            "median": q2,
            "q3": q3,
            "average": np.average(stat_values),
            "max": np.max(stat_values),
            "std": np.std(stat_values, ddof=0),
            "IQR": iqr,
            "skewness": skew(stat_values),
            "kurtosis": kurtosis(stat_values)  # excess kurtosis (normal = 0)
        }

    return result


def print_stats(detailed_stats: dict):
    """
    Print detailed statistics in formatted output.

    This function iterates over the provided dictionary of detailed statistics and prints each
    statistical category along with its corresponding metrics. Each metric value is formatted to three decimal places.

    Parameters:
      detailed_stats (dict): A dictionary mapping statistical categories (str) to dictionaries where
                             each key is a metric name (str) and each value is a numerical statistic (float or int).
    """
    for stat_key, stat_result_dict in detailed_stats.items():
        for result_key, result_value in stat_result_dict.items():
            print(f"{stat_key} {result_key}: {result_value:.3f}")
       
