"""Parse UWyo sounding text blocks into numeric arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List
import math

import numpy as np

COLUMNS = [
    "PRES",
    "HGHT",
    "TEMP",
    "DWPT",
    "RELH",
    "MIXR",
    "DRCT",
    "SKNT",
    "THTA",
    "THTE",
    "THTV",
]


def _clean_value(value: str) -> float:
    text = value.strip()
    if not text:
        return math.nan
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return math.nan


def _slice_row(line: str) -> List[str]:
    padded = line.rstrip("\r\n")
    while len(padded) < 70:
        padded += " "
    fixed = [padded[i : i + 7] for i in range(0, 70, 7)]
    tail = padded[70:].strip()
    fixed.append(tail)
    return fixed[: len(COLUMNS)]


def importfile(
    filename: str | Path, start_row: int = 10, end_row: int | float | None = None
) -> Dict[str, np.ndarray]:
    """
    Read the sounding text block and convert to numeric arrays.

    Parameters
    ----------
    filename : Path-like
        HTML file produced by download_ua.
    start_row, end_row : ints
        1-indexed bounds on the lines to process, mirroring MATLAB's Import Tool.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from column name to numeric array (NaNs where parsing failed).
    """

    filename = Path(filename)
    with filename.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    start_idx = max(start_row - 1, 0)
    if end_row is None or (isinstance(end_row, float) and math.isinf(end_row)):
        end_idx = len(lines)
    else:
        end_idx = min(int(end_row), len(lines))

    rows = []
    for raw_line in lines[start_idx:end_idx]:
        if not raw_line.strip():
            continue
        slices = _slice_row(raw_line)
        if len(slices) < len(COLUMNS):
            continue
        rows.append([_clean_value(value) for value in slices])

    if not rows:
        raise ValueError("No data rows were parsed from the file.")

    matrix = np.array(rows, dtype=float)
    return {column: matrix[:, idx] for idx, column in enumerate(COLUMNS)}


__all__ = ["importfile", "COLUMNS"]
