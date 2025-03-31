import os
import hashlib
import json

# The file to be used for logging files to determine if a file was modified or new
LOG_FILE = "file_log.json"


def _compute_file_hash(path):
    """
    Compute the SHA256 hash of a file's contents.

    Parameters:
      path (str): The file path to compute the hash for.

    Returns:
      str|None: The hexadecimal SHA256 hash if the file exists, or None if the file is not found.
    """
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        return None


def _load_log():
    """
    Load the log data from the JSON log file.

    Returns:
      dict: A dictionary containing logged file hashes, or an empty dictionary if the log file does not exist.
    """
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_log(log_data):
    """
    Save the log data to the JSON log file.

    Parameters:
      log_data (dict): The dictionary containing file log data to be saved.

    Returns:
      None
    """
    with open(LOG_FILE, "w") as f:
        json.dump(log_data, f, indent=4)


def is_logged(path) -> bool:
    """
    Check if the file is logged and its content remains unchanged.

    This function computes the current SHA256 hash of the file and compares it with the logged hash.

    Parameters:
      path (str): The file path to check.

    Returns:
      bool: True if the file is in the log and its hash matches the logged hash, False otherwise.
    """
    log_data = _load_log()
    if path not in log_data:
        return False

    current_hash = _compute_file_hash(path)
    return log_data[path]["hash"] == current_hash


def log_file(path) -> None:
    """
    Log the file by storing its current SHA256 hash in the log file.

    Parameters:
      path (str): The file path to log.

    Returns:
      None
    """
    log_data = _load_log()
    file_hash = _compute_file_hash(path)

    if file_hash is not None:
        log_data[path] = {"hash": file_hash}
        _save_log(log_data)
