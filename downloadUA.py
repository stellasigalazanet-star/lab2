"""Utilities for downloading University of Wyoming sounding files."""

from __future__ import annotations

import calendar
from pathlib import Path
from typing import List, Tuple
import time
import urllib.request


def _timestamp_token(year: int, month: int, day: int, hour: int) -> str:
    """Return ``YYYYMMDD_HHZ`` for clearer file naming."""
    return f"{year:04d}{month:02d}{day:02d}_{hour:02d}Z"


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

    The output filename follows ``RWS_<station>_<YYYYMMDD_HHZ>.htm`` so each
    day/hour slot maps to a unique file.

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
    timestamp = _timestamp_token(year, month, day, hour)
    filename = outdir / f"RWS_{station_str}_{timestamp}.htm"

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


def download_month(
    region: str,
    station: int | str,
    year: int,
    month: int,
    *,
    outdir: str | Path = ".",
    min_bytes: int = 500,
    retry_pause: int = 100,
) -> Tuple[List[Path], List[Tuple[int, int, Exception]]]:
    """
    Download every available 12Z sounding for the requested month.

    Returns
    -------
    tuple
        Two-element tuple ``(successes, failures)``; each failure is
        ``(day, 12, exception)`` so the caller can retry specific slots.
    """

    _, num_days = calendar.monthrange(year, month)
    successes: List[Path] = []
    failures: List[Tuple[int, int, Exception]] = []
    launch_hour = 12

    for day in range(1, num_days + 1):
        try:
            path = download_ua(
                region,
                station,
                year,
                month,
                day,
                launch_hour,
                outdir=outdir,
                min_bytes=min_bytes,
                retry_pause=retry_pause,
            )
        except Exception as exc:
            failures.append((day, launch_hour, exc))
        else:
            successes.append(path)

    return successes, failures


__all__ = ["download_ua", "download_month"]
