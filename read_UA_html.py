"""Replicates the MATLAB helper that locates the data block inside UWyo HTML files."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple


def read_ua_html(file_path: str | Path) -> Tuple[int, int]:
    """
    Locate the start and end row (1-indexed) for the sounding data table.

    The logic mirrors the MATLAB implementation:
    - Scan for the first line that contains the substring '----' at least four times.
      The sounding data start five lines after that marker.
    - Scan for the line containing 'Station information'. The last data row is the
      line immediately before that marker.

    Returns
    -------
    (start_row, end_row) : tuple of ints
        1-indexed positions suitable for slicing text files in MATLAB style.
    """

    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    start_marker = None
    for idx, line in enumerate(lines):
        count = line.count("----")
        if count > 3:
            start_marker = idx
            break

    if start_marker is None:
        raise ValueError("Unable to locate header delimiter '----'.")

    start_row = start_marker + 1 + 4  # MATLAB counted header lines, then +5

    end_marker = None
    for idx, line in enumerate(lines):
        if "Station information" in line:
            end_marker = idx
            break

    if end_marker is None:
        raise ValueError("Unable to locate 'Station information' footer.")

    end_row = end_marker - 1  # MATLAB subtracts one additional line

    return start_row, end_row


__all__ = ["read_ua_html"]
