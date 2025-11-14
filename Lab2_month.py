"""Generate a July-long temperature diagram from UWyo soundings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")  # ensure plotting works without a display

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from downloadUA import download_month
from importfile import importfile
from read_UA_html import read_ua_html


@dataclass
class MonthlyDiagramConfig:
    station: int = 16546
    region: str = "europe"
    year: int = 2025
    month: int = 7
    hours: Iterable[int] = (0, 12)
    outdir: Path = Path("data/july")
    min_bytes: int = 500
    retry_pause: int = 60
    max_height_km: float = 20.0
    height_step_m: float = 250.0


def _build_temperature_grid(
    profiles: List[Tuple[int, int, Path]],
    cfg: MonthlyDiagramConfig,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, Exception]]]:
    """Interpolate each profile onto a uniform height grid."""

    height_grid = np.arange(
        0.0,
        cfg.max_height_km * 1000.0 + cfg.height_step_m,
        cfg.height_step_m,
        dtype=float,
    )

    columns = len(profiles)
    temps_interp = np.full((height_grid.size, columns), np.nan)
    parse_failures: List[Tuple[int, int, Exception]] = []

    for col, (day, hour, path) in enumerate(profiles):
        try:
            start_row, end_row = read_ua_html(path)
            sounding = importfile(path, start_row, end_row)
        except Exception as exc:  # pragma: no cover - informational logging
            parse_failures.append((day, hour, exc))
            continue

        heights = sounding["HGHT"]
        temps = sounding["TEMP"]
        mask = np.isfinite(heights) & np.isfinite(temps)
        if np.count_nonzero(mask) < 2:
            parse_failures.append((day, hour, ValueError("Insufficient valid samples")))
            continue

        heights_valid = heights[mask]
        temps_valid = temps[mask]
        order = np.argsort(heights_valid)
        heights_sorted = heights_valid[order]
        temps_sorted = temps_valid[order]

        in_range = (heights_sorted >= height_grid[0]) & (heights_sorted <= height_grid[-1])
        heights_clipped = heights_sorted[in_range]
        temps_clipped = temps_sorted[in_range]
        if heights_clipped.size < 2:
            parse_failures.append((day, hour, ValueError("Not enough levels within plotting range")))
            continue

        temps_interp[:, col] = np.interp(
            height_grid,
            heights_clipped,
            temps_clipped,
            left=np.nan,
            right=np.nan,
        )

    return height_grid, temps_interp, parse_failures


def generate_monthly_temperature_diagram(cfg: MonthlyDiagramConfig) -> Tuple[Path, List[Tuple[int, int, Exception]]]:
    """Download July soundings and render a time–height temperature diagram."""

    cfg.outdir.mkdir(parents=True, exist_ok=True)
    successes, download_failures = download_month(
        cfg.region,
        cfg.station,
        cfg.year,
        cfg.month,
        hours=cfg.hours,
        outdir=cfg.outdir,
        min_bytes=cfg.min_bytes,
        retry_pause=cfg.retry_pause,
    )

    if not successes:
        raise RuntimeError("No soundings were downloaded; cannot build diagram.")

    successes.sort(key=lambda entry: (entry[0], entry[1]))
    height_grid, temps_interp, parse_failures = _build_temperature_grid(successes, cfg)

    timestamps = [datetime(cfg.year, cfg.month, day, hour) for day, hour, _ in successes]
    time_axis = mdates.date2num(timestamps)

    fig, ax = plt.subplots(figsize=(12, 6))
    mesh = ax.pcolormesh(
        time_axis,
        height_grid / 1000.0,
        temps_interp,
        shading="auto",
        cmap="coolwarm",
    )
    ax.set_title(f"Station {cfg.station:05d} – Temperature profiles ({cfg.year}-{cfg.month:02d})")
    ax.set_ylabel("Height (km)")
    ax.set_xlabel("Date / Time (UTC)")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%HZ"))
    fig.colorbar(mesh, ax=ax, label="Temperature (°C)")
    fig.tight_layout()

    figure_path = cfg.outdir / f"temperature_diagram_{cfg.year}{cfg.month:02d}.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)

    return figure_path, download_failures + parse_failures


def main() -> None:
    cfg = MonthlyDiagramConfig()
    figure_path, issues = generate_monthly_temperature_diagram(cfg)
    print(f"Saved monthly temperature diagram to {figure_path}")
    if issues:
        print("Some soundings could not be processed:")
        for day, hour, exc in issues:
            print(f"  {day:02d} at {hour:02d}Z -> {exc}")


if __name__ == "__main__":
    main()
