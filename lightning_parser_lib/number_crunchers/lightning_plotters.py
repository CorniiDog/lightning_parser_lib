import pandas as pd
import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
import os
import multiprocessing
from tqdm import tqdm
from io import BytesIO
from PIL import Image
import imageio
from PIL import Image
import re
from typing import Tuple
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from .toolbox import tprint
from . import toolbox

global_shutdown_event = None

def init_worker(shutdown_ev):
    global global_shutdown_event
    global_shutdown_event = shutdown_ev

def plot_strikes_over_time(
    bucketed_strikes_indices_sorted: list[list[int]],
    events: pd.DataFrame,
    output_filename="strike_points_over_time.png",
    _export_fig=True
):
    """
    Generate a scatter plot of lightning strike points over time and save it as an image.

    The function extracts the start time (as a timezone-aware datetime) and the number
    of strike points from each bucket in the sorted list of lightning strikes.
    It then creates a scatter plot with lines connecting the points using Plotly,
    and finally saves the plot to a specified file.

    Parameters:
      bucketed_strikes_indices_sorted (list of list of int): Sorted list of lightning strike indices,
                                                             where each sublist corresponds to a strike.
      events (pandas.DataFrame): DataFrame containing lightning event data, including a 'time_unix' column.
      output_filename (str): The filename to save the resulting plot. Defaults to "strike_points_over_time.png".
      _export_fig (bool): Whether to export the figure to file.
      
    Returns:
      go.Figure: The generated Plotly figure.
    """
    # Prepare data: For each bucket, extract the start time (as a timezone-aware datetime) and the number of strike points.
    plot_data = []
    for strike in bucketed_strikes_indices_sorted:
        start_time_unix = events.iloc[strike[0]]["time_unix"]
        dt = datetime.datetime.fromtimestamp(start_time_unix, tz=datetime.timezone.utc)
        plot_data.append({"Time": dt, "Strike Points": len(strike)})

    if len(plot_data) == 0:
        tprint("No Data Found for Plotting")
        return None

    df_plot = pd.DataFrame(plot_data)
    # Sort the DataFrame by time.
    df_plot.sort_values(by="Time", inplace=True)

    # Compute global start time (earliest strike bucket) for display.
    global_start_time = df_plot["Time"].min().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Create a scatter plot with lines connecting the points.
    fig = px.scatter(
        df_plot,
        x="Time",
        y="Strike Points",
        title=f"Number of Strike Points Over Time ({global_start_time})",
        template="plotly_white",
        labels={"Time": "Time (UTC)", "Strike Points": "Number of Strike Points"},
    )
    fig.update_traces(
        mode="lines+markers",
        marker=dict(size=3, color="red"),
        line=dict(color="darkblue", width=2),
    )
    fig.update_layout(
        title_font_size=18,
        xaxis=dict(showgrid=True, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridcolor="lightgray"),
        margin=dict(l=50, r=50, t=80, b=50),
    )

    if _export_fig:
        # Save as svg
        fig.write_image(output_filename, scale=3)

    return fig


def plot_avg_power_map(
    strike_indices: list[int],
    events: pd.DataFrame,
    lat_bins: int = 500,
    lon_bins: int = 500,
    sigma: float = 1.0,
    transparency_threshold:float = 0.01,
    output_filename: str = "strike_avg_power_map.png",
    _export_fig=True,
    _range=None,
    _bar_range=None,
    _use_start_time=True,
    _explicit_time_unix: float=None
):
    """
    Generate a heatmap of average power (in dBW) over latitude/longitude for a specified lightning strike.

    The function bins the strike event data into a 2D grid and calculates the mean power in each bin.
    It then applies a Gaussian blur to smooth the data and creates a heatmap using Plotly.

    Parameters:
      strike_indices (list of int): List of indices corresponding to rows in the 'events' DataFrame for a specific strike.
      events (pandas.DataFrame): DataFrame containing at least 'lat', 'lon', and 'power_db' columns.
      lat_bins (int): Number of bins for latitude. Defaults to 500.
      lon_bins (int): Number of bins for longitude. Defaults to 500.
      sigma (float): Standard deviation for the Gaussian kernel used for smoothing. Defaults to 1.0.
      transparency_threshold (float): Below this power threshold, the data becomes transparent (set to NaN).
      output_filename (str): Filename for the output image. Defaults to "strike_avg_power_map.png".
      _export_fig (bool): Whether to export the figure to file.
      _range (list or None): Optional axis ranges as [[lat_min, lat_max], [lon_min, lon_max]].
      _bar_range (list or None): Optional color bar range.
      _use_start_time (bool): Whether to use the first event's time as the title reference.

    Returns:
      Tuple[go.Figure, float]: The generated Plotly figure and the maximum value of the blurred statistic.
    """
    if len(strike_indices) == 0:
        return None, None

    strike_events = events.iloc[strike_indices]

    start_unix = strike_events.iloc[0]["time_unix"]
    end_unix = strike_events.iloc[-1]["time_unix"]

    # Get the strike's start time from the first event.
    if not _explicit_time_unix:
        if _use_start_time:
            title_unix = start_unix
        else:
            title_unix = end_unix
    else:
        title_unix = _explicit_time_unix
    title_dt = datetime.datetime.fromtimestamp(title_unix, tz=datetime.timezone.utc)
    frac = int(title_dt.microsecond / 10000)  # Convert microseconds to hundredths (0-99)
    title_str = title_dt.strftime(f"%Y-%m-%d %H:%M:%S.{frac:02d} UTC")

    # Extract lat, lon, and power for binning.
    lat = strike_events["lat"].values
    lon = strike_events["lon"].values
    power = strike_events["power_db"].values

    # Determine the min/max for lat/lon.
    lat_min, lat_max = lat.min(), lat.max()
    lon_min, lon_max = lon.min(), lon.max()

    # Use binned_statistic_2d to compute mean power in each lat/lon bin.
    stat, lat_edges, lon_edges, _ = binned_statistic_2d(
        lat,
        lon,
        power,
        statistic="mean",
        bins=[lat_bins, lon_bins],
        range=_range or [[lat_min, lat_max], [lon_min, lon_max]],
    )

    # Replace NaNs with 0 (or any default) so they appear in the heatmap.
    stat_filled = np.nan_to_num(stat, nan=0.0)

    # Apply Gaussian blur to smooth the binned data.
    # Increase or decrease `sigma` depending on how much smoothing you want.
    blurred_stat = gaussian_filter(stat_filled, sigma=sigma)

    # Remove areas below the transparency threshold by masking them as NaN.
    blurred_stat = np.where(blurred_stat < transparency_threshold, np.nan, blurred_stat)

    # Compute bin centers for plotting.
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    if _bar_range:
        _bar_min = _bar_range[0]
        _bar_max = _bar_range[1]
        _zauto = False
    else:
        _bar_min = None
        _bar_max = None
        _zauto = True


    # Create heatmap trace; x-axis = longitude, y-axis = latitude.
    heatmap = go.Heatmap(
        x=lon_centers,
        y=lat_centers,
        z=blurred_stat,
        colorscale="ice",
        colorbar=dict(title="Average Power (dBW)"),
        zauto=_zauto,
        zmin = _bar_min,
        zmax = _bar_max,
        reversescale = False
    )

    # Build the figure with layout settings.
    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title=f"Smoothed (Gaussian) Average Power Heatmap (dBW) ({title_str})",
        xaxis=dict(title="Longitude", showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title="Latitude", showgrid=True, gridcolor="lightgray"),
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )

    # Export the figure to file (SVG/PNG).
    if _export_fig:
        fig.write_image(output_filename, scale=3)

    return fig, np.nanmax(blurred_stat)

def generate_strike_gif(
    strike_indices: list[int],
    events: pd.DataFrame,
    lat_bins: int = 500,
    lon_bins: int = 500,
    sigma: float = 1.0,
    num_frames: int = 30,
    transparency_threshold: float = 0.01,
    output_filename: str = "strike_power_map_animation.gif",
    duration: float = 3000,
    looped:bool = True,
) -> str:
    """
    Generate a GIF animation of a lightning strike heatmap evolving over a specified number of frames.

    The function sorts the strike events by time, divides them into cumulative segments based on a time interval,
    generates a heatmap for each segment, converts each plot to an image, and compiles them into a GIF.

    Parameters:
      strike_indices (list of int): List of indices for lightning strike events.
      events (pandas.DataFrame): DataFrame containing at least 'lat', 'lon', 'power_db', and 'time_unix' columns.
      lat_bins (int): Number of bins for latitude. Defaults to 500.
      lon_bins (int): Number of bins for longitude. Defaults to 500.
      sigma (float): Standard deviation for the Gaussian kernel used for smoothing. Defaults to 1.0.
      num_frames (int): Number of frames in the resulting GIF. Defaults to 30.
      transparency_threshold (float): Power threshold for transparency masking.
      output_filename (str): Filename for the output GIF.
      duration (float): Total duration of the GIF in milliseconds. Defaults to 3000.
      looped (bool): Whether the GIF should loop indefinitely.

    Returns:
      str: The filename where the GIF animation is saved.
    """

    # Preprocess: sort indices by time and extract corresponding times into a NumPy array.
    sorted_indices = sorted(strike_indices, key=lambda idx: events.loc[idx, "time_unix"])
    sorted_times = np.array([events.loc[idx, "time_unix"] for idx in sorted_indices])

    # Determine the overall time span among the selected events.
    min_time = events.loc[sorted_indices[0], "time_unix"]
    max_time = events.loc[sorted_indices[-1], "time_unix"]
    time_interval = (max_time - min_time) / num_frames

    strike_events = events.iloc[strike_indices]

    # Extract lat, lon, and power for binning.
    lat = strike_events["lat"].values
    lon = strike_events["lon"].values

    # Determine the min/max for lat/lon.
    lat_min, lat_max = lat.min(), lat.max()
    lon_min, lon_max = lon.min(), lon.max()

    _range = [[lat_min, lat_max], [lon_min, lon_max]]
    # Use binned_statistic_2d to compute mean power in each lat/lon bin.

    _, max_stat = plot_avg_power_map(
            sorted_indices,
            events,
            lat_bins=lat_bins,
            lon_bins=lon_bins,
            sigma=sigma,
            _export_fig=False,
            _range=_range,
            transparency_threshold=transparency_threshold,
            _use_start_time=False
        )
    
    if max_stat == None: # If no data is returned
        tprint("Returning???")
        return
    
    start_time_unix = strike_events.iloc[0]["time_unix"]
    end_time_unix = strike_events.iloc[-1]["time_unix"]

    frames = []
    # Generate frames based on time intervals.
    for frame in range(1, num_frames + 1):
        current_time_threshold = min_time + frame * time_interval
        # Filter events up to the current time threshold.
        
        # Quickly find the cutoff position using np.searchsorted.
        pos = np.searchsorted(sorted_times, current_time_threshold, side='right')
        frame_indices = sorted_indices[:pos]   
        fig, _ = plot_avg_power_map(
            frame_indices,
            events,
            lat_bins=lat_bins,
            lon_bins=lon_bins,
            sigma=sigma,
            transparency_threshold=transparency_threshold,
            _export_fig=False,
            _range=_range,
            _bar_range=[0, max_stat],
            _use_start_time=False,
            _explicit_time_unix=toolbox.lerp(start_time_unix, end_time_unix, (frame/num_frames))
        )
        
        # Convert the Plotly figure to an image.
        img_bytes = fig.to_image(format="png", scale=3)
        img = Image.open(BytesIO(img_bytes))
        frames.append(np.array(img))

    # Logic to set to 0 (means indefinitely)
    # Else loop once
    looped = 0 if looped else 1

    # Split the gif's duration to the number of frames
    frame_duration = duration/num_frames

    # Save all frames as a GIF.
    imageio.mimsave(output_filename, frames, duration=frame_duration, loop=looped)
    return output_filename

def _plot_strike(args):
    """
    Helper function to generate and save an average power heatmap for a single lightning strike.
    
    Designed for parallel processing, this function unpacks its arguments, determines a safe filename
    based on the strike's start time, and calls the appropriate plotting function to generate and save the image or GIF.

    Parameters:
      args (tuple): Contains:
          - strike_indices (list of int): Indices for a lightning strike.
          - events (pandas.DataFrame): DataFrame with lightning event data.
          - strike_dir (str): Directory to save the generated image/GIF.
          - as_gif (bool): Whether to export as a GIF instead of a static image.
          - sigma (float): Gaussian kernel standard deviation for smoothing.
          - transparency_threshold (float): Threshold for applying transparency.
          
    Returns:
      None
    """
    strike_indices, events, strike_dir, as_gif, sigma, transparency_threshold = args

    if global_shutdown_event and global_shutdown_event.is_set():
        return

    # Get the start time
    start_time_unix = events.iloc[strike_indices[0]]["time_unix"]
    start_time_dt = datetime.datetime.fromtimestamp(
        start_time_unix, tz=datetime.timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")

    safe_start_time = re.sub(r'[<>:"/\\|?*]', '_', str(start_time_dt))

    if not as_gif:
        output_filename = os.path.join(strike_dir, f"{safe_start_time}.png")
        plot_avg_power_map(strike_indices, events, output_filename=output_filename, sigma=sigma, transparency_threshold=transparency_threshold)
    else:
        output_filename = os.path.join(strike_dir, f"{safe_start_time}.gif")
        generate_strike_gif(strike_indices, events, output_filename=output_filename, sigma=sigma, transparency_threshold=transparency_threshold)


def plot_all_strikes(
    bucketed_strike_indices, events, strike_dir="strikes", num_cores=1, as_gif=False, sigma=1.0, transparency_threshold=0.01
):
    """
    Generate and save heatmaps for all detected lightning strikes using parallel processing.

    This function prepares argument tuples for each lightning strike and utilizes a multiprocessing pool
    to generate average power heatmaps concurrently, displaying a progress bar during processing.

    Parameters:
      bucketed_strike_indices (list of list of int): List of strike groups (each group is a list of event indices).
      events (pandas.DataFrame): DataFrame containing lightning event data.
      strike_dir (str): Directory to save the generated images/GIFs. Defaults to "strikes".
      num_cores (int): Number of worker processes to use. Defaults to 1.
      as_gif (bool): Whether to export as a GIF instead of static images.
      sigma (float): Gaussian kernel standard deviation for smoothing. Defaults to 1.0.
      transparency_threshold (float): Threshold for transparency application.
            
    Returns:
      None
    """

    shutdown_event = multiprocessing.Event()

    # Prepare the argument tuples for each strike
    args_list = [
        (strike_indices, events, strike_dir, as_gif, sigma, transparency_threshold)
        for strike_indices in bucketed_strike_indices
    ]
    
    try:
        if num_cores > 1:
            # Use a pool of worker processes to parallelize
            with multiprocessing.Pool(processes=num_cores, initializer=init_worker, initargs=(shutdown_event,)) as pool:
                # Use imap so that we can attach tqdm for a progress bar
                for _ in tqdm(pool.imap(_plot_strike, args_list), total=len(args_list)):
                    pass
        else: # use only one core, in-line
            for args in args_list:
                _plot_strike(args)
    except KeyboardInterrupt:
        shutdown_event.set()


def plot_lightning_stitch(
    lightning_correlations: list[Tuple[int, int]], 
    events: pd.DataFrame,
    output_filename: str = "strike_stitched_map.png",
    _export_fig: bool = True,
    _dimensions: list[list[str, str], list[str, str]] = (["lon", "Longitude"], ["lat", "Latitude"]),
    _range=None,
    _use_start_time = True,
    _explicit_time_unix: float = None
    
) -> go.Figure:
    """
    Plot stitched lightning correlations on a 2D scatter plot using latitude and longitude.

    Each correlation (parent-child pair) is plotted as a line whose color represents the elapsed time 
    since the earliest event. A dummy trace is used to display a continuous colorbar.
    Invisible points enforce the specified axis ranges.

    Parameters:
      lightning_correlations (list[Tuple[int, int]]): List of tuples representing lightning event correlations.
      events (pandas.DataFrame): DataFrame with event data, including "lat", "lon", "time_unix", and "power_db".
      output_filename (str): Filename for exporting the plot image.
      _export_fig (bool): Whether to export the figure as an image.
      _dimensions (list): Specifies the data field and header for x and y axes. Defaults to longitude and latitude.
      _range (list or None): Optional axis ranges as [[lat_min, lat_max], [lon_min, lon_max]]. If None, computed from data.
      _use_start_time (bool): Determines whether to use the start or end event time for the plot title.

    Returns:
      go.Figure: The generated Plotly figure containing the stitched lightning plot.
    """

    # Skip if no data
    if len(lightning_correlations) == 0:
        return

    # Get the start time from the last correlation's child event.
    start_time_unix = events.loc[lightning_correlations[0][0]]["time_unix"]
    end_time_unix = events.loc[lightning_correlations[-1][-1]]["time_unix"]

    # Filter the events dataframe for all points between the two timestamps.
    points_between = events[(events["time_unix"] >= start_time_unix) & (events["time_unix"] <= end_time_unix)]

    # Get the strike's start time from the first event.
    if not _explicit_time_unix:
        if _use_start_time:
            marker_time_unix = start_time_unix
        else:
            marker_time_unix = end_time_unix
    else:
        marker_time_unix = _explicit_time_unix
    
    marker_time_dt = datetime.datetime.fromtimestamp(marker_time_unix, tz=datetime.timezone.utc)

    frac = int(marker_time_dt.microsecond / 10000)  # Convert microseconds to hundredths (0-99)
    marker_time_str = marker_time_dt.strftime(f"%Y-%m-%d %H:%M:%S.{frac:02d} UTC")

    plot_range = _range
    computed_lat_min, computed_lat_max, computed_lon_min, computed_lon_max = None, None, None, None
    avg_unixes = []  # List to store average unix time for each segment

    x_dim: str = _dimensions[0][0] or "lon" # i.e. "lon"
    x_header: str = _dimensions[0][1] or "Longitude"
    y_dim: str = _dimensions[1][0] or "lat" # i.e. "lat"
    y_header: str = _dimensions[1][1] or "Latitude"

    # First pass: compute plot range and collect average unix times.
    for parent_idx, child_idx in lightning_correlations:
        parent_row = events.loc[parent_idx]
        child_row = events.loc[child_idx]
        x1, y1 = parent_row[x_dim], parent_row[y_dim]
        x2, y2 = child_row[x_dim], child_row[y_dim]

        # Update computed range
        for x_val in [x1, x2]:
            if computed_lon_min is None or x_val < computed_lon_min:
                computed_lon_min = x_val
            if computed_lon_max is None or x_val > computed_lon_max:
                computed_lon_max = x_val
        for y_val in [y1, y2]:
            if computed_lat_min is None or y_val < computed_lat_min:
                computed_lat_min = y_val
            if computed_lat_max is None or y_val > computed_lat_max:
                computed_lat_max = y_val

        # Compute the average unix time for the segment.
        avg_unix = np.average([parent_row["time_unix"], child_row["time_unix"]])
        avg_unixes.append(avg_unix)

    if plot_range is None:
        plot_range = [[computed_lat_min, computed_lat_max], [computed_lon_min, computed_lon_max]]

    # Determine the seconds offset and range for color scaling.
    unix_offset = min(avg_unixes)
    max_diff = max(avg_unixes) - unix_offset  # total elapsed seconds

    # Create individual line traces for each correlation segment.
    line_traces = []
    unique_indices = set()
    for i, (parent_idx, child_idx) in enumerate(lightning_correlations):
        parent_row = events.loc[parent_idx]
        child_row = events.loc[child_idx]
        unique_indices.add(parent_idx)
        unique_indices.add(child_idx)
        # Calculate seconds after the earliest event.
        seconds_after = np.average([parent_row["time_unix"], child_row["time_unix"]]) - unix_offset
        # Compute interpolation factor t (0 = earliest, 1 = latest).
        t = seconds_after / max_diff if max_diff else 0.5
        color = sample_colorscale('Cividis', t)[0]
        trace = go.Scatter(
            x=[parent_row[x_dim], child_row[x_dim]],
            y=[parent_row[y_dim], child_row[y_dim]],
            mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
            hoverinfo="none"
        )
        line_traces.append(trace)

    # Dummy trace to display the "Seconds After Start" colorbar.
    dummy_trace = go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            colorscale='Cividis',
            cmin=0,
            cmax=max_diff,
            color=[0, max_diff],
            colorbar=dict(title="Time (s)", tickformat="0.2f", x=1.0)
        ),
        showlegend=False,
        hoverinfo="none"
    )

    # Extract coordinates for the strike points.
    points_x = []
    points_y = []
    point_powers = []
    for idx in unique_indices:
        row = events.loc[idx]
        points_x.append(row[x_dim])
        points_y.append(row[y_dim])
        point_powers.append(row["power_db"])

    # Create a scatter trace for the strike points with a second colorbar for power_db.
    points_trace = go.Scatter(
        x=points_x,
        y=points_y,
        mode="markers",
        marker=dict(
            size=2,
            color=point_powers,
            colorscale='Agsunset', 
            colorbar=dict(title="Power (dBW)", x=1.15)  # Adjust x to position second colorbar
        ),
        name="Lightning Strikes",
        showlegend=False
    )

    # Create background trace: plot all points between the two Unix timestamps with marker size 1.
    background_trace = go.Scatter(
        x=points_between[x_dim],
        y=points_between[y_dim],
        mode="markers",
        marker=dict(size=2, color="black"),
        opacity= 0.5,
        showlegend=False,
        hoverinfo="skip"
    )

    # Add invisible points to enforce the specified range.
    invisible_trace = go.Scatter(
        x=[plot_range[1][0], plot_range[1][1]],
        y=[plot_range[0][0], plot_range[0][1]],
        mode="markers",
        marker=dict(opacity=0),
        showlegend=False,
        hoverinfo="none"
    )

    # Combine all traces.
    fig = go.Figure(data=[background_trace] + line_traces + [dummy_trace, points_trace, invisible_trace])
    fig.update_layout(
        title=f"Lightning Strike Stitching ({marker_time_str})",
        xaxis=dict(title=f"{x_header}", range=plot_range[1], showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title=f"{y_header}", range=plot_range[0], showgrid=True, gridcolor="lightgray"),
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50)
    )

    if _export_fig:
        fig.write_image(output_filename, scale=3)

    return fig, plot_range





def plot_lightning_stitch_gif(
    lightning_correlations: list[Tuple[int, int]], 
    events: pd.DataFrame,
    num_frames: int = 30,
    output_filename: str = "strike_stitched_map_animation.gif",
    duration: float = 3000,
    looped: bool = True,
) -> str:
    """
    Generate a GIF animation of the lightning stitching process.

    The function progressively adds lightning correlations and creates an animated plot showing
    the incremental construction of the stitched lightning map.

    Parameters:
      lightning_correlations (list[Tuple[int, int]]): List of correlation tuples.
      events (pandas.DataFrame): DataFrame containing event data with "lat", "lon", and "time_unix".
      num_frames (int): Number of frames in the GIF animation. Defaults to 30.
      output_filename (str): Filename for the output GIF.
      duration (float): Total duration (in milliseconds) of the GIF. Defaults to 3000.
      looped (bool): Whether the GIF should loop indefinitely.

    Returns:
      str: The filename where the GIF animation is saved.
    """

    # Sort the correlations by the child's event time to ensure proper progression.
    sorted_correlations = sorted(lightning_correlations, key=lambda corr: events.loc[corr[1], "time_unix"])
    
    # Compute the full plot range from all correlations if not provided.
    computed_lat_min, computed_lat_max, computed_lon_min, computed_lon_max = None, None, None, None
    for parent_idx, child_idx in lightning_correlations:
        parent_row = events.loc[parent_idx]
        child_row = events.loc[child_idx]
        for lat_val in [parent_row["lat"], child_row["lat"]]:
            if computed_lat_min is None or lat_val < computed_lat_min:
                computed_lat_min = lat_val
            if computed_lat_max is None or lat_val > computed_lat_max:
                computed_lat_max = lat_val
        for lon_val in [parent_row["lon"], child_row["lon"]]:
            if computed_lon_min is None or lon_val < computed_lon_min:
                computed_lon_min = lon_val
            if computed_lon_max is None or lon_val > computed_lon_max:
                computed_lon_max = lon_val
    full_range = [[computed_lat_min, computed_lat_max], [computed_lon_min, computed_lon_max]]
    
    frames = []
    total_corr = len(sorted_correlations)

    start_time_unix = events.loc[lightning_correlations[0][0]]["time_unix"]
    end_time_unix = events.loc[lightning_correlations[-1][-1]]["time_unix"]

    # Generate frames by progressively adding more correlations.
    for frame in range(1, num_frames + 1):
        # Determine the cutoff index for the current frame (ensure at least one correlation is shown).
        cutoff = max(1, int(round((frame / num_frames) * total_corr)))
        subset = sorted_correlations[:cutoff]
        
        # Generate the plot for the current subset.
        # Note: _export_fig is False so the figure is not saved to disk.
        fig, _ = plot_lightning_stitch(
            subset, 
            events, 
            output_filename="temp.png",  # Dummy filename; image export is disabled.
            _export_fig=False,
            _range=full_range,
            _use_start_time=False,
            _explicit_time_unix=toolbox.lerp(start_time_unix, end_time_unix, (frame/num_frames))
        )
        
        # Convert the Plotly figure to an image.
        img_bytes = fig.to_image(format="png", scale=3)
        img = Image.open(BytesIO(img_bytes))
        frames.append(np.array(img))
    
    # Calculate frame duration (in milliseconds) and set loop parameter (0 for infinite looping).
    frame_duration = duration / num_frames
    loop_val = 0 if looped else 1
    
    # Save all frames as a GIF animation.
    imageio.mimsave(output_filename, frames, duration=frame_duration, loop=loop_val)
    return output_filename

            
def _plot_strike_stitchings(args):
    """
    Helper function to generate and save a stitched lightning map for a single group of correlations.

    This function unpacks its input arguments, constructs a safe filename based on the last correlation's child event time,
    and then generates either a static image or a GIF animation of the stitched lightning map.

    Parameters:
      args (tuple): Contains:
          - lightning_correlations (list[Tuple[int, int]]): Correlation tuples for a group.
          - events (pandas.DataFrame): DataFrame with lightning event data.
          - output_dir (str): Directory to save the output.
          - as_gif (bool): If True, generate a GIF; otherwise, a static image.
    
    Returns:
      None.
    """
    lightning_correlations, events, output_dir, as_gif = args

    if global_shutdown_event and global_shutdown_event.is_set():
        return

    if len(lightning_correlations) == 0:
        return

    # Use the last correlation's child event time for filename generation.
    start_time_unix = events.loc[lightning_correlations[-1][1]]["time_unix"]
    start_time_dt = datetime.datetime.fromtimestamp(start_time_unix, tz=datetime.timezone.utc)
    start_time_str = start_time_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    safe_start_time = re.sub(r'[<>:"/\\|?*]', '_', start_time_str)

    if not as_gif:
        output_filename = os.path.join(output_dir, f"{safe_start_time}.png")
        # Generate a static stitched map.
        plot_lightning_stitch(
            lightning_correlations,
            events,
            output_filename=output_filename,
            _export_fig=True,
            _use_start_time=False
        )
    else:
        output_filename = os.path.join(output_dir, f"{safe_start_time}.gif")
        # Generate a GIF animation of the stitching process.
        plot_lightning_stitch_gif(
            lightning_correlations,
            events,
            output_filename=output_filename
        )


def plot_all_strike_stitchings(
    bucketed_lightning_correlations: list[list[Tuple[int, int]]],
    events: pd.DataFrame,
    output_dir: str = "strike_stitchings",
    num_cores: int = 1,
    as_gif: bool = False
):
    """
    Generate and save stitched lightning maps for all groups of correlations using parallel processing.

    This function prepares argument tuples for each group of lightning correlations and uses a multiprocessing pool
    to generate stitched maps concurrently. A progress bar displays the processing status.

    Parameters:
      bucketed_lightning_correlations (list of list of Tuple[int, int]]): Each sublist contains correlation tuples for a group.
      events (pandas.DataFrame): DataFrame with lightning event data.
      output_dir (str): Directory where output images/GIFs will be saved. Defaults to "strike_stitchings".
      num_cores (int): Number of worker processes to use. Defaults to 1.
      as_gif (bool): If True, output as GIF animations; otherwise, static images.

    Returns:
      None.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    shutdown_event = multiprocessing.Event()

    args_list = [
        (lightning_correlations, events, output_dir, as_gif)
        for lightning_correlations in bucketed_lightning_correlations
    ]

    try:
        if num_cores > 1:
            with multiprocessing.Pool(processes=num_cores, initializer=init_worker, initargs=(shutdown_event,)) as pool:
                for _ in tqdm(pool.imap(_plot_strike_stitchings, args_list), total=len(args_list)):
                    pass
        else: #Only use one core, in-line
            for args in args_list:
                _plot_strike_stitchings(args)
    except KeyboardInterrupt:
        shutdown_event.set()



