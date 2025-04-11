"""
Lightning Strike Visualization Module
---------------------------------------

This module synthesizes and visualizes lightning strike data using Datashader and Matplotlib.
It generates synthetic lightning events and produces multi-panel plots that display various aspects
of the data (e.g. altitude vs. time, latitude vs. longitude, etc.), including overlaid stitching lines
that connect sequential events. The final visualization is cropped and saved as a TIFF file.

Functions:
    main() -> None
        Runs the example pipeline: generates synthetic data, creates a strike image, and saves the result.
    colormap_to_hex(cmap_name: str) -> List[str]
        Converts a Matplotlib colormap into a list of hex color codes for use with Datashader.
    forceAspect(ax, aspect: float = 1.0) -> None
        Adjusts the aspect ratio of a Matplotlib axis to match a specified width-to-height ratio.
    conditional_formatter_factory(min_val, max_val, max_decimal_places: int = 4) -> callable
        Returns a formatter function to format axis tick labels based on the range of values.
    custom_time_formatter(x, pos) -> str
        Formats a numeric time value (Matplotlib date number) into a human-readable HH:MM:SS string,
        including microseconds if present.
    range_bufferize(list_items: list[float], l_buffer_extension: float) -> Tuple[float, float]
        Computes a buffered range for a list of numerical values based on a given buffer extension.
        
Classes:
    XLMAParams
        A parameter container for configuring the lightning strike visualization.
    
Functions (continued):
    create_strike_image(xlma_params: XLMAParams, events: pd.DataFrame,
                        strike_indeces: List[int],
                        strike_stitchings: List[Tuple[int, int]]) -> Image
        Generates the complete lightning strike image with multiple subplots and optional stitching lines.
"""
import matplotlib
matplotlib.use(backend='Agg')  # Use non-GUI backend suitable for multiprocessing
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import lnglat_to_meters
from datashader.transfer_functions import spread
from matplotlib import colormaps, rcParams
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import scipy
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.dates as mdates
import datetime
from PIL import Image
from matplotlib.collections import LineCollection
import io
import imageio
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm
import re
import os
import multiprocessing
from . import toolbox

def main():
    """
    Main entry point of the module.

    Generates a synthetic DataFrame of lightning events, constructs stitching indices,
    configures the visualization parameters, creates the final strike image using those parameters,
    and saves the cropped image as a TIFF file.
    """
    print(list(colormaps))

    ######################################################################
    # Test Bench dataframe
    ######################################################################
    num_pts = 10000
    np.random.seed(42)

    # Base coordinates (e.g., near central Texas)
    base_lat = 32.0
    base_lon = -97.74

    # Create a random walk for a lightning strike’s horizontal path.
    # Typical step size (in degrees) is chosen to simulate ~100 m changes.
    step_std = 0.001  # Roughly 111 m per 0.001° latitude; longitude steps are similar near Texas

    lat_steps = np.random.normal(0, step_std, num_pts)
    lon_steps = np.random.normal(0, step_std, num_pts)

    lats = base_lat + np.cumsum(lat_steps)
    lons = base_lon + np.cumsum(lon_steps)

    # Simulate altitude: descending from cloud base (~4000 m) to ground (0 m)
    alts = np.linspace(4000, 0, num_pts)
    # Add some variability
    alt_noise = np.random.normal(0, 50, num_pts)  # 50 m noise
    alts = alts + alt_noise
    alts[alts < 0] = 0  # Clamp to ground level

    # Simulate time over a short duration (e.g., lightning occurs over ~1 second)
    time_unix = 1649574956 + np.linspace(0, 1, num_pts)

    # Simulate power: around -70 dBW with some variability typical in lightning signals.
    power_db = np.random.normal(-70, 5, num_pts)

    # Build the realistic DataFrame
    events = pd.DataFrame({
        'lat': lats,
        'lon': lons,
        'power_db': power_db,
        'alt': alts,
        'time_unix': time_unix
    })

    strike_indeces = [i for i in range(num_pts)]

    strike_stitchings = [(i, i + 1) for i in range(num_pts - 1)]

    xlma_params = XLMAParams()
    xlma_params.dark_theme = True

    strike_img, _ = create_strike_image(xlma_params, events, strike_indeces, strike_stitchings)

    export_strike_image(strike_img, "strike.tiff")

    strike_gif_buffer, _ = create_strike_gif(xlma_params, events, strike_indeces, strike_stitchings)
    
    export_strike_gif(strike_gif_buffer, "strike.gif")

    # Convert cropped_image to a NumPy array for plotting
    img_array = np.array(strike_img)

    # Display the image using matplotlib
    plt.imshow(img_array)
    plt.axis('off')  # Hide the axis
    plt.show()

    plt.clf()

global_shutdown_event = None
def init_worker(shutdown_ev):
    global global_shutdown_event
    global_shutdown_event = shutdown_ev

def colormap_to_hex(cmap_name: str) -> List[str]:
    """
    Convert a Matplotlib colormap to a list of hex color strings.

    Parameters:
        cmap_name (str): Name of the Matplotlib colormap to convert.

    Returns:
        List[str]: A list of hex string representations for 256 resampled colors.

    References:
      - https://matplotlib.org/stable/users/explain/colors/colormaps.html
      - https://matplotlib.org/stable/gallery/color/colormap_reference.html
    """
    cmap = colormaps[cmap_name].resampled(256)  # Resample to 256 colors
    return [mcolors.rgb2hex(cmap(i)) for i in range(cmap.N)]

def forceAspect(ax, aspect: float = 1.0):
    """
    Force the aspect ratio of a Matplotlib axis.

    Adjusts the aspect ratio (width/height) of the axis 'ax' so that the plotted data
    preserves the desired ratio. Uses either linear or logarithmic scaling based on the axis.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to adjust.
        aspect (float, optional): The desired width-to-height ratio. Defaults to 1.0.

    References:
      - https://stackoverflow.com/a/45123239
    """
    #aspect is width/height
    scale_str = ax.get_yaxis().get_scale()
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    if scale_str=='linear':
        asp = abs((xmax-xmin)/(ymax-ymin))/aspect
    elif scale_str=='log':
        asp = abs((scipy.log(xmax)-scipy.log(xmin))/(scipy.log(ymax)-scipy.log(ymin)))/aspect
    ax.set_aspect(asp)

def conditional_formatter_factory(min_val, max_val, max_decimal_places: int = 4):
    """
    Creates a formatter function for axis ticks based on data range.

    This factory returns a function that determines the number of decimal places
    to use when formatting numbers for ticks, based on the overall range of data.

    Parameters:
        min_val (float): Minimum value in the data range.
        max_val (float): Maximum value in the data range.
        max_decimal_places (int, optional): Maximum number of decimal places to display.
            Defaults to 4.

    Returns:
        callable: A formatter function that formats numbers accordingly.
    """
    def formatter(x, pos):
        diff = abs(max_val - min_val)
        if diff == 0.0:
            return f"{x:.{1}f}"
        
        if diff > 10:
            decimal_places = 0
        elif diff > 1:
            decimal_places = 1
        else:
            decimal_places = min(len(str(int(1/diff))) + 1, max_decimal_places)
        return f"{x:.{decimal_places}f}"

    return formatter

def custom_time_formatter(x, pos) -> str:
    """
    Format a numeric time value into a readable time string.

    Converts a Matplotlib date number 'x' into a string formatted as HH:MM:SS with
    optional microseconds if present.

    Parameters:
        x (float): The numeric time value.
        pos: Unused positional parameter required by FuncFormatter.

    Returns:
        str: The formatted time string.
    """
    dt = mdates.num2date(x)  # Convert axis number to datetime
    s = dt.strftime('%H:%M:%S')  # Base HH:MM:SS string
    if dt.microsecond:
        # Convert microseconds to fractional seconds, format up to 6 decimals, and remove trailing zeros.
        frac = dt.microsecond / 1e6
        frac_str = f"{frac:.6f}"[1:].rstrip('0')
        s += frac_str
    return s

def range_bufferize(list_items: list[float], l_buffer_extension: float) -> Tuple[float, float]:
    """
    Compute a buffered numerical range from a list of values.

    Calculates a lower and upper bound that extends beyond the min and max of the input
    list by a percentage defined by l_buffer_extension.

    Parameters:
        list_items (List[float]): A list of numerical values.
        l_buffer_extension (float): The fractional extension to apply to the range.

    Returns:
        Tuple[float, float]: The buffered (min, max) range.
    """
    l_min, l_max = min(list_items), max(list_items)
    l_buffer_size = max(abs(l_max - l_min), 0.2) * l_buffer_extension
    return (l_min - l_buffer_size, l_max + l_buffer_size)

######################################################################
# Parameters
######################################################################

class XLMAParams:
    """
    Parameter container for configuring lightning strike visualizations.

    This class encapsulates configuration parameters for a comprehensive lightning strike visualization pipeline,
    incorporating options for data processing, rendering resolution, and stylistic control for both Datashader and
    Matplotlib components. The parameters determine how raw lightning event data is processed and depicted across
    multiple subplots, covering aspects such as temporal evolution, altitude profiles, spatial distributions, and
    aggregated statistics. They govern not only the graphical representation—such as colormap conversion, dynamic point 
    spreading, and plotting resolution—but also the overlaying of stitching lines and geographical boundaries (e.g., 
    counties, cities) using external shapefiles.
        
    """

    def __init__(self,
            time_as_datetime: bool = True,
            points_resolution_multiplier: int = 5,
            max_pixel_size: int = 1,
            altitude_group_size: int = 100,
            altitude_graph_max_pixel_size: int = 1,
            altitude_graph_line_thickness: float = 0.5,
            altitude_graph_alpha = 0.5,
            altitude_graph_resolution_multiplier: int = 1,
            buffer_extension: float = 0.1,
            stitching_line_thickness: float = 0.5,
            stitching_alpha: float = 0.5,
            colormap_scheme: str = "rainbow",
            font_size: int = 7,
            dark_theme: bool = True,
            time_unit: str = 'time_unix',
            alt_unit: str = 'alt',
            x_unit: str = 'lon',
            y_unit: str = 'lat',
            color_unit: str = 'time_unix',
            zero_time_unit_if_color_unit: bool = True,
            zero_colorbar: bool = False,
            num_pts_unit: str = 'num_pts',
            alt_group_unit: str = "alt_group",
            dpi: int = 300,
            title: str = "LYLOUT LMA",
            figure_size: Tuple[int, int] = (7, 7),
            cartopy_paths: List[str] = None,
            tiger_path: str = None,
            county_line_alpha=0.1,
            county_line_width=1,
            county_text_font_size=3,
            county_text_color='lime',
            county_text_alpha=1,
            headers: Dict[str, str] = None):
        """
        Initialize the XLMAParams instance with visualization parameters.

        Parameters:
            time_as_datetime (bool): If True, interpret time data as datetime objects; otherwise, treat as numeric timestamps.
            points_resolution_multiplier (int): Multiplier to scale the resolution of point plots in Datashader.
            max_pixel_size (int): Maximum pixel size for Datashader's dynamic spreading effect.
            altitude_group_size (int): Interval used to group altitude values for aggregation and visualization.
            altitude_graph_max_pixel_size (int): Maximum pixel size applied to the aggregated altitude graph.
            altitude_graph_line_thickness (float): Line thickness used to connect points in the altitude graph.
            altitude_graph_alpha (float): Transparency level for altitude graph stitching lines and overlays.
            altitude_graph_resolution_multiplier (int): Resolution multiplier specific to the altitude plot to adjust detail.
            buffer_extension (float): Fractional extension to the data range to add as padding on the plot axes.
            stitching_line_thickness (float): Line thickness for stitching lines connecting sequential lightning events.
            stitching_alpha (float): Transparency level for stitching lines between events.
            colormap_scheme (str): Name of the Matplotlib colormap to convert into hex color codes for Datashader.
            font_size (int): Font size for all plot text elements, including labels and annotations.
            dark_theme (bool): If True, applies a dark background theme to the plots.
            time_unit (str): Column name for the time values in the lightning event DataFrame.
            alt_unit (str): Column name for altitude values in the event DataFrame.
            x_unit (str): Column name for the x-axis coordinate (typically longitude) in the event DataFrame.
            y_unit (str): Column name for the y-axis coordinate (typically latitude) in the event DataFrame.
            color_unit (str): Column name used for determining color mapping of the events; may be the same as time_unit.
            zero_time_unit_if_color_unit (bool): If True and time is used for color mapping, adjusts the time values to start at zero.
            zero_colorbar (bool): If True, forces the colorbar to start at zero regardless of the data range.
            num_pts_unit (str): Identifier for the number of points metric, used in aggregated statistical plots.
            alt_group_unit (str): Label for altitude grouping applied during data aggregation.
            dpi (int): Dots per inch resolution for saving the output visualization.
            title (str): The title of the visualization to be rendered on the final image.
            figure_size (Tuple[int, int]): Dimensions (width, height in inches) of the resulting figure.
            cartopy_paths (List[str]): List of file paths to geographical boundary shapefiles (e.g., counties, cities) for overlaying on the plot.
            tiger_path (str): File path to TIGER/Line shapefiles used for detailed geographical features.
            county_line_alpha (float): Transparency level for drawn county boundary lines.
            county_line_width (float): Line width used when rendering county boundaries.
            county_text_font_size (int): Font size for county name labels on the map.
            county_text_color (str): Color for county name annotations; can be a hex code (e.g., "#39FF14" for neon green) or a named color.
            county_text_alpha (float): The alpha for showing text
            headers (Dict[str, str]): Dictionary mapping data column names to human-readable header labels for axes and legends.
        """

        self.time_as_datetime = time_as_datetime
        self.points_resolution_multiplier = points_resolution_multiplier
        self.max_pixel_size = max_pixel_size
        self.altitude_group_size = altitude_group_size
        self.altitude_graph_max_pixel_size = altitude_graph_max_pixel_size
        self.altitude_graph_line_thickness = altitude_graph_line_thickness
        self.altitude_graph_alpha = altitude_graph_alpha
        self.altitude_graph_resolution_multiplier = altitude_graph_resolution_multiplier
        self.buffer_extension = buffer_extension
        self.stitching_line_thickness = stitching_line_thickness
        self.stitching_alpha = stitching_alpha
        self.colormap_scheme = colormap_scheme
        self.font_size = font_size
        self.dark_theme = dark_theme
        self.time_unit = time_unit
        self.alt_unit = alt_unit
        self.x_unit = x_unit
        self.y_unit = y_unit
        self.color_unit = color_unit
        self.zero_time_unit_if_color_unit = zero_time_unit_if_color_unit
        self.zero_colorbar = zero_colorbar
        self.num_pts_unit = num_pts_unit
        self.alt_group_unit = alt_group_unit
        self.dpi = dpi
        self.title = title
        self.figure_size = figure_size
        self.cartopy_paths = cartopy_paths
        self.tiger_path = tiger_path
        self.county_line_alpha=county_line_alpha
        self.county_line_width=county_line_width
        self.county_text_font_size=county_text_font_size
        self.county_text_color=county_text_color
        self.county_text_alpha = county_text_alpha
        
        # Default headers
        self.headers = {
            'lat': 'Latitude',
            'lon': 'Longitude',
            'alt': 'Altitude (m)',
            'power_db': 'Power Logarithmic (dBW)',
            'time_unix': 'Time (s)',
            'num_pts': 'Number of Points',
            'datetime': 'Time (UTC)',
            'reduced_chi2': 'Reduced Chi^2',
            'num_stations': 'Number of Stations',
            'power': 'Power (W)',
            'mask': 'Hexidecimal Bitmask',
            'stations': 'Stations Contributed',
            'x': 'Meters (ECEF X WGS84)',
            'y': 'Meters (ECEF Y WGS84)',
            'z': 'Meters (ECEF Z WGS84)',
            'file_name': 'File Name',
            'id': 'Row Identification'
        }
        # Replace or Add new Headers
        if headers:
            for key, value in headers:
                self.headers[key] = value

class RangeParams:
    """
    Container for numerical ranges used in visualization axis limits and normalization.

    This class encapsulates the various buffered numerical ranges (e.g., time, altitude,
    spatial coordinates, and colorbar values) that are computed and applied to plotting axes
    for consistent scaling and presentation.

    Attributes:
        time_unit_range (tuple or list, optional): The buffered range for the time unit values.
        time_unit_datetime_range (tuple or list, optional): The buffered range as datetime values.
        time_range (tuple or list, optional): The raw range of time values.
        alt_range (tuple or list, optional): The buffered range of altitude values.
        x_range (tuple or list, optional): The buffered range of x-coordinate values (e.g., longitude).
        y_range (tuple or list, optional): The buffered range of y-coordinate values (e.g., latitude).
        num_pts_range (tuple or list, optional): The buffered range for the number of points (used in aggregation).
        colorbar_range (tuple or list, optional): The range of values used for colorbar normalization.
    """

    def __init__(self,
                 time_unit_range=None,
                 time_unit_datetime_range=None,
                 time_range=None,
                 alt_range=None,
                 x_range=None,
                 y_range=None,
                 num_pts_range=None,
                 colorbar_range=None):
        """
        Initialize a new instance of the RangeParams class.

        Parameters:
            time_unit_range (tuple or list, optional): The buffered range for the time unit values. Defaults to None.
            time_unit_datetime_range (tuple or list, optional): The buffered range as datetime values. Defaults to None.
            time_range (tuple or list, optional): The raw range of time values. Defaults to None.
            alt_range (tuple or list, optional): The buffered range of altitude values. Defaults to None.
            x_range (tuple or list, optional): The buffered range of x-coordinate values (e.g., longitude). Defaults to None.
            y_range (tuple or list, optional): The buffered range of y-coordinate values (e.g., latitude). Defaults to None.
            num_pts_range (tuple or list, optional): The buffered range for the number of points (used in aggregation). Defaults to None.
            colorbar_range (tuple or list, optional): The range of values used for colorbar normalization. Defaults to None.
        """
        self.time_unit_range = time_unit_range
        self.time_unit_datetime_range = time_unit_datetime_range
        self.time_range = time_range
        self.alt_range = alt_range
        self.x_range = x_range
        self.y_range = y_range
        self.num_pts_range = num_pts_range
        self.colorbar_range = colorbar_range


def create_strike_image(xlma_params: XLMAParams,
                        events: pd.DataFrame,
                        strike_indeces: List[int],
                        strike_stitchings: Optional[List[Tuple[int, int]]] = None,
                        range_params: RangeParams = None) -> Tuple[Image.Image, RangeParams]:
    """
    Create a composite lightning strike image with multiple subplots and stitching lines.

    This function generates a multi-panel plot based on the provided lightning event data.
    It processes and aggregates data into various views (e.g., altitude vs. time, latitude vs. longitude, etc.),
    utilizes Datashader for rendering each panel, overlays stitching lines using Matplotlib's LineCollection
    where applicable, and finally crops the rendered image to the desired boundaries. Additionally, it
    computes and updates range parameters for consistent axis scaling and normalization.

    Parameters:
        xlma_params (XLMAParams): An instance of XLMAParams containing visualization parameters.
        events (pd.DataFrame): DataFrame containing the lightning event data.
        strike_indeces (List[int]): A list of event indices to include in the visualization.
        strike_stitchings (List[Tuple[int, int]]): A list of tuples where each tuple represents a pair of indices
            to be connected by a stitching line.
        range_params (RangeParams, optional): An instance of RangeParams with pre-computed ranges.
            If not provided, default ranges will be computed. Defaults to a new RangeParams() instance.

    Returns:
        Tuple[Image.Image, RangeParams]:
            A tuple where the first element is a PIL Image object representing the cropped composite visualization,
            and the second element is the updated RangeParams instance containing all computed ranges.
    """
    plt.close('all')
    plt.clf()
    plt.close()
    df = events.iloc[strike_indeces].copy(deep=True)
    all_x_arr, all_y_arr, all_alt_arr = events[xlma_params.x_unit], events[xlma_params.y_unit], events[xlma_params.alt_unit]

    start_time_unit = df.iloc[0][xlma_params.time_unit]
    start_time = datetime.datetime.fromtimestamp(timestamp=start_time_unit, tz=datetime.timezone.utc)
    start_time_str = start_time.strftime("%d %b %Y")

    description = f"{start_time_str}"

    ######################################################################
    # Initialization and config adjustment
    ######################################################################
        
    if xlma_params.dark_theme:
        plt.style.use('dark_background')  # Apply dark mode
    else:
        plt.style.use(style='default')

    plt.rcParams.update({'font.size': xlma_params.font_size})

    # Convert Matplotlib colormap to hex colors for Datashader
    colormap = colormap_to_hex(xlma_params.colormap_scheme)


    ######################################################################
    # Fig creation and formatting
    ######################################################################
    fig = plt.figure(figsize=xlma_params.figure_size)

    # Create a 2x2 sub-grid with custom ratios
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 4], width_ratios=[4, 1, 0.1], wspace=0)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])  # Top left
    ax2 = fig.add_subplot(gs[1, 1])  # Top right
    ax3 = fig.add_subplot(gs[2, 0])  # Bottom left (largest)
    ax4 = fig.add_subplot(gs[2, 1])  # Bottom right
    ax_colorbar = fig.add_subplot(gs[:, 2])  # Bottom right

    ax0.sharey(ax1)
    ax2.sharey(ax1)

    ax1.sharex(ax3)

    ax3.sharey(ax4)

    ######################################################################
    # Colorbar
    ######################################################################
    color_unit_specific = xlma_params.color_unit + "_cu"
    df[color_unit_specific] = df[xlma_params.color_unit]
    if ((xlma_params.color_unit == xlma_params.time_unit and xlma_params.zero_time_unit_if_color_unit) or xlma_params.zero_colorbar) and len(df) > 0:
        df[color_unit_specific] -= df[xlma_params.color_unit].iloc[0]
    
    if not range_params:
        range_params = RangeParams()
        range_params.time_unit_range = range_bufferize(df[xlma_params.time_unit], xlma_params.buffer_extension)
        range_params.time_unit_datetime_range = pd.to_datetime(range_params.time_unit_range, unit='s', utc=True).to_list()
        range_params.time_range = range_bufferize(df[xlma_params.time_unit], xlma_params.buffer_extension)
        range_params.alt_range = range_bufferize(df[xlma_params.alt_unit], xlma_params.buffer_extension)
        range_params.x_range = range_bufferize(df[xlma_params.x_unit], xlma_params.buffer_extension)
        range_params.y_range = range_bufferize(df[xlma_params.y_unit], xlma_params.buffer_extension)
        
        range_params.colorbar_range = [df[color_unit_specific].min(), df[color_unit_specific].max()]

    time_unit_datetime = 'datetime'
    df['datetime'] = pd.to_datetime(df[xlma_params.time_unit], unit='s', utc=True)

    # Compute normalization based on your data (e.g., power_db values)
    norm = mcolors.Normalize(vmin=range_params.colorbar_range[0], vmax=range_params.colorbar_range[1])
    # Create a ScalarMappable with the chosen colormap
    sm = plt.cm.ScalarMappable(norm=norm, cmap=colormaps[xlma_params.colormap_scheme])
    sm.set_array([])  # Necessary for matplotlib to handle the colorbar correctly

    fig.colorbar(sm, cax=ax_colorbar, orientation='vertical', label=xlma_params.headers[xlma_params.color_unit])


    ######################################################################
    # Alt and Time Plot
    ######################################################################
    # Define canvas using longitude and latitude ranges (in degrees)
    cvs = ds.Canvas(plot_width=295*xlma_params.points_resolution_multiplier, plot_height=50*xlma_params.points_resolution_multiplier,
                    x_range=range_params.time_range,
                    y_range=range_params.alt_range)

    # Aggregate using mean of power_db (for overlapping points)
    agg = cvs.points(df, xlma_params.time_unit, xlma_params.alt_unit, ds.mean(color_unit_specific))
    agg = agg.where(agg != 0)
    # Create Datashader image with dynamic spreading
    img = spread(tf.shade(agg, cmap=colormap, how='linear', span=range_params.colorbar_range), px=xlma_params.max_pixel_size)

    if range_params.time_unit_datetime_range:
        extent = (*range_params.time_unit_datetime_range, *range_params.alt_range)

        ax0.imshow(X=img.to_pil(), extent=extent, origin='lower')
        ax0.set_xlabel(xlma_params.headers[time_unit_datetime])
    else:
        extent = (*range_params.time_range, *range_params.alt_range)
        ax0.imshow(X=img.to_pil(), extent=extent, origin='lower')
        ax0.set_xlabel(xlma_params.headers[time_unit_datetime])
    ax0.set_ylabel(xlma_params.headers[xlma_params.alt_unit])

    ######################################################################
    # Alt and Num Pts Plot
    ######################################################################
    # Alt and Num Pts Plot using LineCollection
    altitudes = {}
    color_units = {}
    for _, row in df.iterrows():
        altitude_group = xlma_params.altitude_group_size * (row[xlma_params.alt_unit] // xlma_params.altitude_group_size)
        if altitude_group not in altitudes:
            altitudes[altitude_group] = 0
            color_units[altitude_group] = []
        altitudes[altitude_group] += 1
        color_units[altitude_group].append(row[color_unit_specific])

    alt_dict = {
        xlma_params.alt_group_unit: [],
        xlma_params.num_pts_unit: []
    }
    alt_dict[color_unit_specific] = []

    for altitude_group, num_pts in sorted(altitudes.items()):
        alt_dict[xlma_params.alt_group_unit].append(altitude_group)
        alt_dict[xlma_params.num_pts_unit].append(num_pts)
        alt_dict[color_unit_specific].append(np.mean(color_units[altitude_group]))

    alt_df = pd.DataFrame(alt_dict)
    range_params.num_pts_range = range_params.num_pts_range or range_bufferize(alt_df[xlma_params.num_pts_unit], xlma_params.buffer_extension)
    alt_df_sorted = alt_df.sort_values(by=xlma_params.alt_group_unit)

    # Instead of using ds.Canvas.line, we create segments from the sorted DataFrame.
    segments = []
    num_pts_values = alt_df_sorted[xlma_params.num_pts_unit].values
    alt_values = alt_df_sorted[xlma_params.alt_group_unit].values
    for i in range(len(alt_df_sorted) - 1):
        segments.append([(num_pts_values[i], alt_values[i]),
                        (num_pts_values[i + 1], alt_values[i + 1])])

    # Choose the marker color based on the theme.
    marker_color = 'black'
    if xlma_params.dark_theme:
        marker_color = 'white'

    # Create and add the LineCollection.
    lc_alt = LineCollection(segments, colors=marker_color,
                            linewidths=xlma_params.altitude_graph_line_thickness,
                            alpha=xlma_params.altitude_graph_alpha)
    ax2.add_collection(lc_alt)

    # Set the axes limits explicitly.
    ax2.set_xlim(range_params.num_pts_range)
    ax2.set_ylim(range_params.alt_range)
    ax2.set_xlabel(xlma_params.headers[xlma_params.num_pts_unit])
    ax2.set_ylabel(f"Chunked \n({xlma_params.headers[xlma_params.alt_unit]}//{xlma_params.altitude_group_size})")

    # Optionally, overlay the Datashader image for the points.
    cvs = ds.Canvas(plot_width=200 * xlma_params.altitude_graph_resolution_multiplier,
                    plot_height=200 * xlma_params.altitude_graph_resolution_multiplier,
                    x_range=range_params.num_pts_range,
                    y_range=range_params.alt_range)
    agg = cvs.points(alt_df, xlma_params.num_pts_unit, xlma_params.alt_group_unit,
                    ds.mean(color_unit_specific))
    img = spread(tf.shade(agg, cmap=colormap, how='linear', span=range_params.colorbar_range),
                    px=xlma_params.altitude_graph_max_pixel_size)
    extent = (*range_params.num_pts_range, *range_params.alt_range)
    ax2.imshow(X=img.to_pil(), extent=extent, origin='upper', zorder=3)
    ax2.set_xlabel(xlma_params.headers[xlma_params.num_pts_unit])
    ax2.set_ylabel(f"Chunked \n(" + xlma_params.headers[xlma_params.alt_unit] + f"//{xlma_params.altitude_group_size})")


    ######################################################################
    # Lat and Lon Plot
    ######################################################################
    # Define canvas using longitude and latitude ranges (in degrees)
    cvs = ds.Canvas(plot_width=200*xlma_params.points_resolution_multiplier, plot_height=200*xlma_params.points_resolution_multiplier,
                    x_range=range_params.x_range,
                    y_range=range_params.y_range)

    # Aggregate using mean of power_db (for overlapping points)
    agg = cvs.points(df, xlma_params.x_unit, xlma_params.y_unit, ds.mean(color_unit_specific))

    # Create Datashader image with dynamic spreading
    img = spread(tf.shade(agg, cmap=colormap, how='linear', span=range_params.colorbar_range), px=xlma_params.max_pixel_size)

    extent = (*range_params.x_range, *range_params.y_range)
    ax3.imshow(X=img.to_pil(), extent=extent, origin='upper', zorder=4)

    if strike_stitchings:
        segments = []
        for stitch in strike_stitchings:
            i1, i2 = stitch
            # Safely retrieve the coordinates from the DataFrame using iloc
            point1 = [all_x_arr[i1], all_y_arr[i1]]
            point2 = [all_x_arr[i2], all_y_arr[i2]]
            segments.append([point1, point2])

        # Create a LineCollection for efficiency with many segments.
        lc = LineCollection(segments, colors=marker_color, linewidths=xlma_params.stitching_line_thickness, alpha=xlma_params.stitching_alpha)
        lc.set_zorder(3)  # Set desired z-order before adding
        ax3.add_collection(lc)

    if xlma_params.cartopy_paths:
        for path in xlma_params.cartopy_paths:
            counties = gpd.read_file(path)

            bounding_box = box(range_params.x_range[0], range_params.y_range[0], range_params.x_range[1], range_params.y_range[1])  # Example: part of South Texas

            # Filter counties that intersect with bounding box
            counties_in_box = counties[counties.geometry.intersects(bounding_box)]

            if len(counties_in_box) == 0:
                continue

            counties_in_box.boundary.plot(ax=ax3, edgecolor=marker_color, linewidth=xlma_params.county_line_width, zorder=2, alpha=xlma_params.county_line_alpha)
            # Add county name labels
            for _, row in counties_in_box.iterrows():
                centroid = row.geometry.centroid
                
                x_buffer_size = (range_params.x_range[1] - range_params.x_range[0]) * xlma_params.buffer_extension
                y_buffer_size = (range_params.y_range[1] - range_params.y_range[0]) * xlma_params.buffer_extension

                in_x_bounds = range_params.x_range[1] - x_buffer_size > centroid.x > range_params.x_range[0] + x_buffer_size
                in_y_bounds = range_params.y_range[1] - y_buffer_size > centroid.y > range_params.y_range[0] + y_buffer_size
                if not in_x_bounds or not in_y_bounds:
                    continue

                ax3.text(centroid.x, centroid.y, row['NAME'], ha='center', color=xlma_params.county_text_color, fontsize=xlma_params.county_text_font_size, alpha=xlma_params.county_text_alpha)

    ax3.set_xlim(extent[0], extent[1])
    ax3.set_ylim(extent[2], extent[3])
    ax3.set_xlabel(xlma_params.headers[xlma_params.x_unit])
    ax3.set_ylabel(xlma_params.headers[xlma_params.y_unit])

    ######################################################################
    # Lat and Alt Plot
    ######################################################################
    # Define canvas using altitude and latitude ranges (in degrees)
    cvs = ds.Canvas(plot_width=50*xlma_params.points_resolution_multiplier, plot_height=200*xlma_params.points_resolution_multiplier,
                    x_range=range_params.alt_range,
                    y_range=range_params.y_range)

    # Aggregate using mean of power_db (for overlapping points)
    agg = cvs.points(df, xlma_params.alt_unit, xlma_params.y_unit, ds.mean(color_unit_specific))

    # Create Datashader image with dynamic spreading
    img = spread(tf.shade(agg, cmap=colormap, how='linear', span=range_params.colorbar_range), px=xlma_params.max_pixel_size)

    extent = (*range_params.alt_range, *range_params.y_range)
    ax4.imshow(X=img.to_pil(), extent=extent, origin='upper', zorder=3)

    if strike_stitchings:
        segments = []
        for stitch in strike_stitchings:
            i1, i2 = stitch
            # Safely retrieve the coordinates from the DataFrame using iloc
            point1 = [all_alt_arr[i1], all_y_arr[i1]]
            point2 = [all_alt_arr[i2], all_y_arr[i2]]
            segments.append([point1, point2])

        # Create a LineCollection for efficiency with many segments.
        lc = LineCollection(segments, colors=marker_color, linewidths=xlma_params.stitching_line_thickness, alpha=xlma_params.stitching_alpha)
        lc.set_zorder(2)  # Set desired z-order before adding
        ax4.add_collection(lc)

    ax4.set_xlabel(xlma_params.headers[xlma_params.alt_unit])

    ######################################################################
    # Alt and Lon Plot
    ######################################################################
    # Define canvas using longitude and altitude ranges (in degrees)
    cvs = ds.Canvas(plot_width=200*xlma_params.points_resolution_multiplier, plot_height=50*xlma_params.points_resolution_multiplier,
                    x_range=range_params.x_range,
                    y_range=range_params.alt_range)

    # Aggregate using mean of power_db (for overlapping points)
    agg = cvs.points(df, xlma_params.x_unit, xlma_params.alt_unit, ds.mean(color_unit_specific))

    # Create Datashader image with dynamic spreading
    img = spread(tf.shade(agg, cmap=colormap, how='linear', span=range_params.colorbar_range), px=xlma_params.max_pixel_size)

    extent = (*range_params.x_range, *range_params.alt_range)
    ax1.imshow(X=img.to_pil(), extent=extent, origin='upper', zorder=3)

    if strike_stitchings:
        segments = []
        for stitch in strike_stitchings:
            i1, i2 = stitch
            # Safely retrieve the coordinates from the DataFrame using iloc
            point1 = [all_x_arr[i1], all_alt_arr[i1]]
            point2 = [all_x_arr[i2], all_alt_arr[i2]]
            segments.append([point1, point2])

        # Create a LineCollection for efficiency with many segments.
        lc = LineCollection(segments, colors=marker_color, linewidths=xlma_params.stitching_line_thickness, alpha=xlma_params.stitching_alpha)
        lc.set_zorder(2)  # Set desired z-order before adding
        ax1.add_collection(lc)

    ax1.set_ylabel(xlma_params.headers[xlma_params.alt_unit])

    ######################################################################
    # Aspect ratio adjustment
    ######################################################################
    forceAspect(ax0, 5.9/1)
    forceAspect(ax1, 4/1)
    forceAspect(ax2, 1)
    forceAspect(ax3, 1)
    forceAspect(ax4, 1/4)

    ######################################################################
    # Time series adjustment
    ######################################################################
    pos0 = ax0.get_position()
    pos1 = ax1.get_position()
    ax0.set_position([pos1.x0, pos0.y0, pos0.width, pos0.height])

    ######################################################################
    # Label modification
    ######################################################################
    fig.align_ylabels([ax0, ax1, ax3])
    fig.align_xlabels([ax3, ax4])

    ax0.xaxis.set_label_coords(0.5, -0.24)
    ax1.xaxis.set_label_coords(0.5, -0.24)
    ax2.xaxis.set_label_coords(0.5, -0.24)

    ######################################################################
    # Axis tick number closeness
    ######################################################################
    padding = 1
    ax0.tick_params(axis='both', pad=padding)
    ax1.tick_params(axis='both', pad=padding)
    ax2.tick_params(axis='both', pad=padding)
    ax3.tick_params(axis='both', pad=padding)
    ax4.tick_params(axis='both', pad=padding)

    ######################################################################
    # Formatting
    ######################################################################
    x_min, x_max = ax0.get_xlim()
    y_min, y_max = ax0.get_ylim()

    if xlma_params.time_as_datetime:
        ax0.xaxis.set_major_formatter(FuncFormatter(custom_time_formatter))
    else:
        ax0.xaxis.set_major_formatter(FuncFormatter(conditional_formatter_factory(x_min, x_max)))
    ax0.yaxis.set_major_formatter(FuncFormatter(conditional_formatter_factory(y_min, y_max)))

    y_min, y_max = ax1.get_ylim()
    ax1.yaxis.set_major_formatter(FuncFormatter(conditional_formatter_factory(y_min, y_max)))

    x_min, x_max = ax4.get_xlim()
    ax4.xaxis.set_major_formatter(FuncFormatter(conditional_formatter_factory(x_min, x_max)))

    x_min, x_max = ax3.get_xlim()
    y_min, y_max = ax3.get_ylim()
    ax3.xaxis.set_major_formatter(FuncFormatter(conditional_formatter_factory(x_min, x_max)))
    ax3.yaxis.set_major_formatter(FuncFormatter(conditional_formatter_factory(y_min, y_max)))

    ######################################################################
    # Titles
    ######################################################################
    fig.suptitle(xlma_params.title, fontsize=12, x=0.55, y=0.95)
    fig.text(0.55, 0.9, description, ha="center", fontsize=9)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=xlma_params.dpi)
    plt.close('all')
    plt.clf()
    plt.close()
    buf.seek(0)

    # Open the image using Pillow
    img = Image.open(buf)
    width, height = img.size

    # Calculate crop boundaries
    left = 250
    top = 50
    right = width - 50
    bottom = height - 100

    # Crop the image (box format: left, top, right, bottom)
    cropped_img = img.crop((left, top, right, bottom))

    return cropped_img, range_params

def _create_strike_gif_utility(args_list):
    frames = []
    for args in args_list:
        if global_shutdown_event and global_shutdown_event.is_set():
            break

        total_events, frame, num_frames, strike_indeces, strike_stitchings, xlma_params, events, range_params = args

        # Determine the cutoff index for the current frame.
        cutoff = max(1, int(round(total_events * frame / num_frames)))
        # Select the subset of event indices.
        partial_indices = strike_indeces[:cutoff]
        # If strike_stitchings is provided, filter for those with both endpoints within the current cutoff.
        if strike_stitchings is not None:
            partial_stitchings = [
                (i1, i2) for (i1, i2) in strike_stitchings if i1 in partial_indices and i2 in partial_indices
            ]
        else:
            partial_stitchings = None

        # Generate the composite strike image for the current subset.
        composite_img, range_params = create_strike_image(
            xlma_params, events, partial_indices, partial_stitchings, range_params
        )
        # Append the frame as a NumPy array.
        frames.append(np.array(composite_img))

    return frames

def create_strike_gif(
    xlma_params: XLMAParams,
    events: pd.DataFrame,
    strike_indeces: List[int],
    strike_stitchings: Optional[List[Tuple[int, int]]] = None,
    num_frames: int = 60,
    duration: int = 6000,
    looped: bool = True,
    range_params: RangeParams = None,
    num_cores:int = 1
) -> Tuple[io.BytesIO, RangeParams]:
    """
    Generate an animated GIF of the composite lightning strike visualization and return
    it as an in-memory byte stream (BytesIO) that is savable.

    Parameters:
        xlma_params (XLMAParams): Parameter container for the lightning strike visualization.
        events (pd.DataFrame): DataFrame containing lightning event data.
        strike_indeces (List[int]): List of indices for the events to include.
        strike_stitchings (Optional[List[Tuple[int, int]]]): Optional list of index pairs for connecting events.
        num_frames (int, optional): Number of frames in the animated GIF (default is 60).
        duration (int, optional): Total duration of the GIF in milliseconds (default is 6000).
        looped (bool, optional): Whether the GIF should loop indefinitely (True) or play once (False).
        range_params (RangeParams, optional): Precomputed or default range parameters for consistent scaling.
        num_cores (int): The number of cores to multiprocess to generate the image faster.

    Returns:
        Tuple[io.BytesIO, RangeParams]:
            - A BytesIO object containing the GIF data.
            - The updated RangeParams instance with computed plotting ranges.
    """
    shutdown_event = multiprocessing.Event()
    # Initialize range parameters by making the first plot (without stitchings).
    _, range_params = create_strike_image(
        xlma_params, events, strike_indeces, None
    )

    frames = []
    total_events = len(strike_indeces)

    args_list = [
        (total_events, frame, num_frames, strike_indeces, strike_stitchings, xlma_params, events, range_params)
        for frame in range(1, num_frames + 1)
    ]

    args_list_bucketed = toolbox.split_into_groups(args_list, num_workers=30)

    try:
        if num_cores > 1:
            with multiprocessing.Pool(processes=num_cores, initializer=init_worker, initargs=(shutdown_event,)) as pool:
                for result in tqdm(pool.imap(_create_strike_gif_utility, args_list_bucketed), total=len(args_list_bucketed), desc="Generating GIF"):
                    frames += result
        else: #Only use one core, in-line
            for args_list in tqdm(args_list_bucketed, desc="Generating GIF"):
                frames += _create_strike_gif_utility(args_list_bucketed)
    except KeyboardInterrupt:
        shutdown_event.set()        

    # Compute duration per frame and set loop parameter (0 means indefinitely loop).
    frame_duration = duration / num_frames
    loop_val = 0 if looped else 1

    # Save the frames as an animated GIF to an in-memory BytesIO stream.
    gif_buffer = io.BytesIO()
    imageio.mimsave(gif_buffer, frames, duration=frame_duration, loop=loop_val, format="GIF")
    gif_buffer.seek(0)  # Reset buffer pointer to the beginning

    return gif_buffer, range_params

def export_strike_image(strike_image: Image, export_path: str):
    """
    Export a lightning strike image to a file.

    This function saves a PIL Image object to the specified file path. The image format is inferred from 
    the file extension (e.g., 'tiff', 'png', 'jpeg') provided in export_path.

    Parameters:
        strike_image (PIL.Image.Image): The PIL Image object representing the lightning strike visualization.
        export_path (str): The file path where the image should be saved.

    Example:
        >>> from PIL import Image
        >>> # Create a simple white image for demonstration.
        >>> image = Image.new('RGB', (100, 100), color='white')
        >>> export_strike_image(image, "output_image.tiff")
    """
    strike_image.save(export_path)


def export_strike_gif(gif_buffer: io.BytesIO, export_path: str):
    """
    Export an animated GIF of a lightning strike to a file.

    This function writes the in-memory GIF data stored in a BytesIO buffer to the specified file path 
    using binary write mode.

    Parameters:
        gif_buffer (io.BytesIO): A BytesIO object containing the GIF data.
        export_path (str): The file path where the animated GIF should be saved.

    Example:
        >>> import io
        >>> # Create a BytesIO buffer with sample GIF data for demonstration.
        >>> gif_data = io.BytesIO(b"GIF87a...")
        >>> export_strike_gif(gif_data, "output_animation.gif")
    """
    with open(export_path, "wb") as f:
        f.write(gif_buffer.getvalue())

def export_stats(xlma_params: XLMAParams, events: pd.DataFrame, bucketed_indeces: List[List[int]], export_path: str):
    """
    Exports a statistics plot to an image file.

    The plot displays event time (UTC) on the x-axis and the number of points per bucket on the y-axis.
    Each non-empty bucket is represented by one point located at the mean time of its events.

    Parameters:
        xlma_params (XLMAParams): Configuration parameters; expects xlma_params.time_unit for the time column name,
                                  xlma_params.time_as_datetime flag, and optionally xlma_params.dpi for resolution.
        events (pd.DataFrame): DataFrame containing lightning event data.
        bucketed_indeces (List[List[int]]): List of buckets, where each bucket is a list of event indices.
        export_path (str): The file path where the generated image will be saved.
    """
    plt.close('all')
    plt.clf()
    plt.close()
    if xlma_params.dark_theme:
        plt.style.use('dark_background')  # Apply dark mode
    else:
        plt.style.use(style='default')

    # Convert the time column to UTC datetime format if necessary.

    times = pd.to_datetime(events[xlma_params.time_unit], unit='s', utc=True)

    agg_times = []
    num_pts = []

    # Iterate over each bucket, computing the representative (mean) time and the number of events.
    for bucket in bucketed_indeces:
        if not bucket:  # Skip empty buckets
            continue
        bucket_times = times.iloc[bucket]
        rep_time = bucket_times.mean()
        agg_times.append(rep_time)
        num_pts.append(len(bucket))

    # Choose the line color based on the theme.
    marker_color = 'black'
    if xlma_params.dark_theme:
        marker_color = 'white'

    # Specify the color for markers (dots)
    dot_color = 'red'

    # Use parameters for point size and line thickness.
    # Multiply point size by a factor to ensure visibility.
    point_size = xlma_params.altitude_graph_max_pixel_size
    line_thickness = xlma_params.altitude_graph_line_thickness

    # Create the plot.
    fig, ax = plt.subplots()
    ax.plot(
        agg_times, num_pts,
        marker='o', linestyle='-',
        color=marker_color,
        markersize=point_size,
        linewidth=line_thickness,
        markerfacecolor=dot_color
    )

    ax.xaxis.set_major_formatter(FuncFormatter(custom_time_formatter))

    ax.yaxis.set_major_formatter(FuncFormatter(conditional_formatter_factory(min(num_pts), max(num_pts))))

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Number of Points")

    start_time_unit = events.iloc[0][xlma_params.time_unit]
    start_time = datetime.datetime.fromtimestamp(timestamp=start_time_unit, tz=datetime.timezone.utc)
    start_time_str = start_time.strftime("%d %b %Y")

    ax.set_title(f"Points Over Time\n{start_time_str}")

    # Automatically format the x-axis for better datetime display.
    fig.autofmt_xdate()

    # Export the plot to an image file with the specified DPI.
    dpi_value = xlma_params.dpi if hasattr(xlma_params, 'dpi') else 300
    fig.savefig(export_path, dpi=dpi_value)
    plt.close('all')
    plt.clf()
    plt.close()

def _export_bulk_to_folder(args):
    events, times, output_dir, xlma_params, bucketed_strike_indices, bucketed_strike_correlations = args
    if global_shutdown_event and global_shutdown_event.is_set():
            return
    
    for i, strike_indeces in enumerate(bucketed_strike_indices):
        if global_shutdown_event and global_shutdown_event.is_set():
            break

        start_time_unix = times[strike_indeces[0]]
        start_time_dt = datetime.datetime.fromtimestamp(
            start_time_unix, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")

        safe_start_time = re.sub(r'[<>:"/\\|?*]', '_', str(start_time_dt))

        file_out_path = os.path.join(output_dir, safe_start_time) + ".tiff"

        if bucketed_strike_correlations and len(bucketed_strike_correlations) > 0:
            strike_correlations = bucketed_strike_correlations[i]
        else:
            strike_correlations = None

        strike_image, _ = create_strike_image(xlma_params=xlma_params, events=events, strike_indeces=strike_indeces, strike_stitchings=strike_correlations)
        export_strike_image(strike_image, file_out_path)

def export_bulk_to_folder(events: pd.DataFrame, output_dir: str, bucketed_strike_indices: List[List[int]], bucketed_strike_correlations: List[Tuple[int, int]] = None, num_cores:int = 1, num_workers: int = 25, xlma_params:XLMAParams = XLMAParams()):
    times = events['time_unix']
    shutdown_event = multiprocessing.Event()

    bucketed_bucketed_strike_indices = toolbox.split_into_groups(bucketed_strike_indices, num_workers)

    if bucketed_strike_correlations:
        bucketed_bucketed_strike_correlations = toolbox.split_into_groups(bucketed_strike_correlations, num_workers)
    else:
        bucketed_bucketed_strike_correlations = [None] * len(bucketed_bucketed_strike_indices)

    lightning_groups = []
    for i, sub_bucketed_strike_indices in enumerate(bucketed_bucketed_strike_indices):
        lightning_groups.append((sub_bucketed_strike_indices, bucketed_bucketed_strike_correlations[i]))
        
    args_list = [
        (events, times, output_dir, xlma_params, bucketed_strike_i, bucketed_strike_corr)
        for bucketed_strike_i, bucketed_strike_corr in lightning_groups
    ]

    try:
        if num_cores > 1:
            with multiprocessing.Pool(processes=num_cores, initializer=init_worker, initargs=(shutdown_event,)) as pool:
                for _ in tqdm(pool.imap(_export_bulk_to_folder, args_list), total=len(args_list), desc="Exporting Strikes"):
                    pass
        else: #Only use one core, in-line
            for args in tqdm(args_list, desc="Exporting Strikes"):
                _export_bulk_to_folder(args)
    except KeyboardInterrupt:
        shutdown_event.set()

# Run example code
if __name__ == '__main__':
    main()