"""Utilities for downloading University of Wyoming sounding files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
import urllib.request


def _datestr_token(year: int, month: int, day: int) -> str:
    """Replicates MATLAB's datestr(datenum(...), 12) -> 'mmmyy' with title case month."""
    return datetime(year, month, day).strftime("%b%y")


def download_ua(
    region: str,
    station: int | str,
    year: int,
    month: int,
    day: int,
    hour: int,
    *,
    outdir: str | Path = ".",
    min_bytes: int = 500,
    retry_pause: int = 100,
) -> Path:
    """
    Download a raw sounding HTML file from the UWyo archive.

    Parameters
    ----------
    region : str
        UWyo region identifier, e.g. 'europe'.
    station : int | str
        Five-digit WMO station code.
    year, month, day, hour : int
        Timestamp for the sounding.
    outdir : Path-like, optional
        Directory where the HTML file will be saved.
    min_bytes : int, optional
        Minimum acceptable file size; retry until exceeded.
    retry_pause : int, optional
        Seconds to wait before retrying when the file is too small.

    Returns
    -------
    pathlib.Path
        Path to the downloaded file.
    """

    station_str = f"{int(station):05d}"
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    date_token = _datestr_token(year, month, day)
    filename = outdir / f"RWS_{station_str}_{date_token}.htm"

    query_time = f"{day:02d}{hour:02d}"
    base_url = "http://weather.uwyo.edu/cgi-bin/sounding"
    url = (
        f"{base_url}?region={region}"
        f"&TYPE=TEXT%3ALIST"
        f"&YEAR={year}"
        f"&MONTH={month:02d}"
        f"&FROM={query_time}"
        f"&TO={query_time}"
        f"&STNM={station_str}"
    )

    while True:
        urllib.request.urlretrieve(url, filename)
        size = filename.stat().st_size
        if size > min_bytes:
            break
        time.sleep(retry_pause)

    return filename


__all__ = ["download_ua"]
