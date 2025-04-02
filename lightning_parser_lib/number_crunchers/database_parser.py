import os
import datetime
import sqlite3
from pyproj import Transformer
import pandas as pd
import sys
import traceback
from .toolbox import tprint
from . import logger 
from . import toolbox

def get_dat_files_paths(lightning_data_folder, data_extension):
    """
    Retrieve the full file paths of all files with a specified extension in a given folder.

    Parameters:
      lightning_data_folder (str): The directory containing the data files.
      data_extension (str): The file extension to filter for (e.g., ".dat").

    Returns:
      list[str]: A list of full paths to the files that match the given extension.

    """
    return [
        os.path.join(lightning_data_folder, f)
        for f in os.listdir(lightning_data_folder)
        if f.endswith(data_extension)
    ]


# Default station mask order (each character represents a station in order)
DEFAULT_STATION_MASK_ORDER = "NMLKJIHGFEDC3A"

# Initialize a transformer to convert from WGS84 (lat,lon,alt in EPSG:4979) to ECEF (EPSG:4978)
transformer = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)


def _decode_station_mask(mask_str, station_mask_order=DEFAULT_STATION_MASK_ORDER):
    """
    Decode a hexadecimal station mask string into a comma-separated list of station names.

    Parameters:
      mask_str (str): The hexadecimal mask string representing active stations.
      station_mask_order (str, optional): Order in which stations are represented.
                                          Defaults to DEFAULT_STATION_MASK_ORDER.

    Returns:
      str: A comma-separated list (all as a single string, explicitly) of station names corresponding to active bits in the mask.
    """
    mask_int = int(mask_str.strip(), 16)
    stations = []
    for i, station in enumerate(station_mask_order):
        if mask_int & (1 << i):
            stations.append(station)
    return ",".join(stations)


def _add_to_database(cursor, event):
    """
    Insert an event record into the 'events' table in the database.

    Parameters:
      cursor (sqlite3.Cursor): Database cursor used to execute SQL statements.
      event (tuple): A tuple containing event data in the following order:
                     (time_unix, lat, lon, alt, reduced_chi2, num_stations, power_db, power, mask, stations, x, y, z, file_name)

    Returns:
      None
    """
    cursor.execute(
        """
        INSERT INTO events (
            time_unix, lat, lon, alt, reduced_chi2, num_stations, power_db, power, mask, stations, x, y, z, file_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        event,
    )


def _create_database_if_not_exist(DB_PATH: str = "lylout_db.db"):
    """
    Create the SQLite database and 'events' table if they do not exist.

    Parameters:
      DB_PATH (str): Path to the SQLite database file. Defaults to "lylout_db.db".

    Returns:
      sqlite3.Connection: Connection object to the SQLite database.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time_unix FLOAT,
            lat FLOAT,
            lon FLOAT,
            alt FLOAT,
            reduced_chi2 FLOAT,
            num_stations INTEGER,
            power_db FLOAT,
            power FLOAT,
            mask TEXT,
            stations TEXT,
            x FLOAT,
            y FLOAT,
            z FLOAT,
            file_name TEXT
        )
    """
    )

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_time_unix ON events(time_unix)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_num_stations ON events(num_stations)")
    conn.commit()
    return conn


def _executesql(query, params=None, DB_PATH="lylout_db.db", fetch=True):
    """
    Execute an SQL query on the specified SQLite database.

    Parameters:
      query (str): The SQL query to execute.
      params (list or tuple, optional): Parameters for the parameterized query. Defaults to None.
      DB_PATH (str): Path to the SQLite database file. Defaults to "lylout_db.db".
      fetch (bool): If True, fetch and return the query results; otherwise, commit changes.

    Returns:
      list[sqlite3.Row]|None: A list of sqlite3.Row objects if fetch is True; otherwise, None.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enables accessing rows as dictionaries.
    cursor = conn.cursor()

    if params is None:
        params = []
    cursor.execute(query, params)

    if fetch:
        results = cursor.fetchall()
        conn.close()
        return results
    else:
        conn.commit()
        conn.close()


def _build_where_clause(filters):
    """
    Construct a SQL WHERE clause from provided filter conditions.

    Filters can be specified as a dictionary or a list. For a dictionary, each key-value pair
    represents a column and its required value using equality. For a list, each element must be
    either a tuple (column, operator, value) or a dictionary with keys "column", "operator", and "value".

    Parameters:
      filters (dict or list): Filter conditions for constructing the WHERE clause.

    Returns:
      tuple: A tuple containing the WHERE clause (as a string) and a list of parameters.
    """
    conditions = []
    params = []

    if isinstance(filters, dict):
        for col, val in filters.items():
            conditions.append(f"{col} = ?")
            params.append(val)
    elif isinstance(filters, list):
        for filt in filters:
            if isinstance(filt, tuple) and len(filt) == 3:
                col, op, val = filt
                conditions.append(f"{col} {op} ?")
                params.append(val)
            elif isinstance(filt, dict):
                col = filt.get("column")
                op = filt.get("operator", "=")
                val = filt.get("value")
                if col is None or val is None:
                    raise ValueError(
                        "Each filter dict must have 'column' and 'value' keys."
                    )
                conditions.append(f"{col} {op} ?")
                params.append(val)
            else:
                raise ValueError(
                    f"Filters must be tuples (column, operator, value) or dicts. {filt}, {type(filt)}"
                )
    else:
        raise ValueError("Filters must be either a dict or a list.")

    clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return clause, params


def query_events(filters, DB_PATH="lylout_db.db"):
    """
    Query the 'events' table in the database using specified filter conditions.

    Parameters:
      filters (dict or list): Filter conditions to apply for querying.
      DB_PATH (str): Path to the SQLite database file. Defaults to "lylout_db.db".

    Returns:
      list: A list of sqlite3.Row objects representing the query results.
    """
    where_clause, params = _build_where_clause(filters)
    query = f"SELECT * FROM events {where_clause}"
    return _executesql(query, params, DB_PATH)


def query_events_as_dataframe(filters, DB_PATH="lylout_db.db"):
    """
    Query the 'events' table and return the results as a pandas DataFrame.

    Parameters:
      filters (dict or list): Filter conditions to apply for querying.
      DB_PATH (str): Path to the SQLite database file. Defaults to "lylout_db.db".

    Returns:
      pandas.DataFrame: DataFrame containing the query results.
    """
    results = query_events(filters, DB_PATH)  # Get results as a list of sqlite3.Row
    df = pd.DataFrame(results, columns=get_headers(DB_PATH))  # Convert to DataFrame
    return df


def get_headers(DB_PATH="lylout_db.db") -> list:
    """
    Retrieve the column names (headers) from the 'events' table in the SQLite database.

    Parameters:
      DB_PATH (str): Path to the SQLite database file. Defaults to "lylout_db.db".

    Returns:
      list[str]: A list of column names from the 'events' table.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(events)")
    headers = [row[1] for row in cursor.fetchall()]
    conn.close()
    return headers


def _parse_dat_extension(lylout_path: str, DB_PATH: str = "lylout_db.db"):
    """
    Parse a .dat file containing LYLOUT data, extract event information, and insert the events into the database.

    The function performs the following:
      - Validates that the file has a .dat extension.
      - Reads the header to optionally override the default station mask order.
      - Extracts the base date from the header (using the format "Data start time: MM/DD/YY HH:MM:SS").
      - Locates the start of the data section marked by "*** data ***".
      - Processes each subsequent line to extract event fields, including conversion from UT seconds to a Unix timestamp,
        conversion of power from dBW to watts, and transformation of geodetic coordinates to ECEF.
      - Inserts each parsed event into the 'events' table of the SQLite database.

    Parameters:
      lylout_path (str): Path to the LYLOUT .dat file.
      DB_PATH (str): Path to the SQLite database file. Defaults to "lylout_db.db".

    Returns:
      None

    Raises:
      Exception: If the file does not have a .dat extension, the base date is not found, or the data section is missing.
    """
    if not lylout_path.lower().endswith(".dat"):
        raise Exception("File must be a .dat file")

    with open(lylout_path, "r") as f:
        lines = f.readlines()

    # Find the beginning of data
    data_start_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("*** data ***"):
            data_start_index = i + 1
            break
    if data_start_index is None:
        raise Exception("Data section not found.")

    # Check for an optional station mask order override in the header
    station_mask_order = DEFAULT_STATION_MASK_ORDER
    for i, line in enumerate(lines):
        if i >= data_start_index:
            break
        if line.startswith("Station mask order:"):
            station_mask_order = line.split("Station mask order:")[1].strip()
            break

    # Extract the base date from header (format: "Data start time: MM/DD/YY HH:MM:SS")
    base_date = None
    for i, line in enumerate(lines):
        if i >= data_start_index:
            break
        if line.startswith("Data start time:"):
            parts_date = line.split("Data start time:")[1].strip()
            base_date = datetime.datetime.strptime(parts_date, "%m/%d/%y %H:%M:%S")
            base_date = base_date.replace(tzinfo=datetime.timezone.utc)
            break
    if base_date is None:
        raise Exception("Base date not found in header.")

    # Detect header order from a "Data:" line, if present
    header_order = None
    for i, line in enumerate(lines):
        if i >= data_start_index:
            break
        if line.strip().startswith("Data:"):
            header_line = line.strip()[len("Data:"):].strip()
            header_order = [h.strip() for h in header_line.split(",")]
            break
    # Fallback default header order matching required fields and expected positions
    if header_order is None:
        header_order = [
            "time (UT sec of day)",
            "lat",
            "lon",
            "alt(m)",
            "reduced chi^2",
            "P(dBW)",
            "mask",
        ]
    header_indices = {name: idx for idx, name in enumerate(header_order)}

    # Create the database and events table if they don't exist
    conn = _create_database_if_not_exist(DB_PATH)
    cursor = conn.cursor()

    # Extract filename from lylout path
    file_name = os.path.basename(lylout_path)

    # Process each data line
    for line in lines[data_start_index:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < len(header_order):
            continue  # Skip incomplete lines
        try:
            ut_sec = float(parts[header_indices["time (UT sec of day)"]])
            lat = float(parts[header_indices["lat"]])
            lon = float(parts[header_indices["lon"]])
            alt = float(parts[header_indices["alt(m)"]])
            reduced_chi2 = float(parts[header_indices["reduced chi^2"]])
            power_db = float(parts[header_indices["P(dBW)"]])
            mask_str = parts[header_indices["mask"]]
        except KeyError as e:
            raise Exception(f"Required header field missing: {e}")

        # Convert UT seconds (since midnight UTC) to Unix timestamp
        midnight = base_date.replace(hour=0, minute=0, second=0, microsecond=0)
        event_time = midnight + datetime.timedelta(seconds=ut_sec)
        time_unix = event_time.timestamp()

        # Decode the station bitmask using the (possibly overridden) station_mask_order
        stations_list = _decode_station_mask(mask_str, station_mask_order)

        num_stations = len(stations_list.split(","))

        # Convert geodetic coordinates to ECEF using pyproj
        x, y, z = transformer.transform(lon, lat, alt)
        
        event = (
            time_unix,
            lat,
            lon,
            alt,
            reduced_chi2,
            num_stations,
            power_db,
            10 ** (power_db / 10),  # Conversion from dBW to linear watts
            mask_str,
            stations_list,
            x,
            y,
            z,
            file_name
        )
        
        
        try:
            _add_to_database(cursor, event)
        except Exception as e:
            tprint("Issue with adding data:", event)
            traceback.print_exc(file=sys.stdout)

    conn.commit()
    conn.close()


def parse_lylout(lylout_path: str, DB_PATH: str = "lylout_db.db"):
    """
    Parse a LYLOUT data file and populate the database with event records.

    This function determines if the provided file is a .dat file and, if so, calls the appropriate parser to process and store the data.

    Parameters:
      lylout_path (str): Path to the LYLOUT data file.
      DB_PATH (str): Path to the SQLite database file. Defaults to "lylout_db.db".

    Returns:
      None
    """
    if lylout_path.lower().endswith(".dat"):
        _parse_dat_extension(lylout_path, DB_PATH)

def cache_and_parse_database(cache_dir: str, lightning_data_folder: str, data_extension: str, DB_PATH: str, CACHE_PATH: str):
    """
    Cache and parse lightning data files, updating the SQLite database if changes are detected.

    This function checks whether the contents of the specified lightning data folder have been cached.
    If not cached, it retrieves all files with the specified extension and processes each file by checking
    if it has been logged (indicating previous processing). For any unlogged file, it parses the file to
    update the database and then logs the file to avoid redundant processing. After processing, it updates
    the cache to reflect the current state of the lightning data folder. If no changes are detected, it skips
    reprocessing to save time.

    Parameters:
      cache_dir (str): Directory path where the cache log file is stored.
      lightning_data_folder (str): Directory containing the lightning data files.
      data_extension (str): Extension used to filter the data files (e.g., ".dat").
      DB_PATH (str): Path to the SQLite database file.
      CACHE_PATH (str): Path to the cache file used to track processed data.

    Returns:
      None
    """
    logger.LOG_FILE = os.path.join(cache_dir, "file_log.json")
    if not toolbox.is_cached(lightning_data_folder, CACHE_PATH):
        tprint("New data changed. Updating database")
        dat_file_paths = get_dat_files_paths(lightning_data_folder, data_extension)
        for file_path in dat_file_paths:
            # If the file is not already processed into the SQLite database
            if not logger.is_logged(file_path):
                tprint(file_path, "not appropriately added to the database. Adding...")
                parse_lylout(file_path, DB_PATH)
                logger.log_file(file_path)  # Log the file for no redundant re-processing into the database
            else:
                tprint(file_path, "was parsed and added to the database already")
        toolbox.save_cache_quick(lightning_data_folder, CACHE_PATH)
    else:
        tprint("Nothing changed with the database. Saving time...")
