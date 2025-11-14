"""Python port of Lab2_day.m for plotting UWyo soundings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from downloadUA import download_ua
from importfile import importfile
from read_UA_html import read_ua_html


@dataclass
class SoundingConfig:
    station: int = 16546
    region: str = "europe"
    year: int = 2025
    month: int = 7
    day: int = 1
    hour: int = 12

    @property
    def plot_title(self) -> str:
        return f"Station {self.station:05d} – {self.year}-{self.month:02d}-{self.day:02d} {self.hour:02d}Z"


def _safe_gradient(values: np.ndarray, heights: np.ndarray, top: int, bottom: int) -> float:
    if max(top, bottom) >= len(values) or max(top, bottom) >= len(heights):
        return np.nan
    if heights[top] == heights[bottom]:
        return np.nan
    return (values[top] - values[bottom]) / (heights[top] - heights[bottom])


def _polyfit_subset(x: np.ndarray, y: np.ndarray, up_to_height: float) -> Tuple[np.ndarray, np.ndarray]:
    mask_index = int(np.nanargmin(np.abs(x - up_to_height)))
    if mask_index < 4:
        raise ValueError("Not enough samples for a 4th-order fit.")
    coeffs = np.polyfit(x[: mask_index + 1], y[: mask_index + 1], 4)
    fitted = np.polyval(coeffs, x[: mask_index + 1])
    return fitted[:-1], x[:mask_index]


def _find_tropopause_index(temps: np.ndarray, heights: np.ndarray, window_m: float = 2000.0) -> int | None:
    """
    Return the index of the first level that satisfies the WMO lapse-rate definition.

    The average lapse rate between the candidate level and all levels within the next
    `window_m` meters must not exceed 2 °C/km.
    """

    temps = np.asarray(temps, dtype=float)
    heights = np.asarray(heights, dtype=float)
    mask = np.isfinite(temps) & np.isfinite(heights)
    if np.count_nonzero(mask) < 2:
        return None

    valid_indices = np.flatnonzero(mask)
    valid_temps = temps[mask]
    valid_heights = heights[mask]

    for local_idx, base_height in enumerate(valid_heights):
        trailing_heights = valid_heights[local_idx + 1 :]
        trailing_temps = valid_temps[local_idx + 1 :]
        if trailing_heights.size == 0:
            break
        deltas = trailing_heights - base_height
        within_window = np.where((deltas > 0) & (deltas <= window_m))[0]
        if within_window.size == 0:
            continue
        lapse_rates = -(trailing_temps[within_window] - valid_temps[local_idx]) / (deltas[within_window] / 1000.0)
        if np.all(lapse_rates <= 2.0):
            return int(valid_indices[local_idx])
    return None


def _tropospheric_lapse_rate(temps: np.ndarray, heights: np.ndarray, trop_index: int | None) -> float | None:
    """Return -ΔT/Δz (K/km) from the surface to the tropopause level."""

    if trop_index is None:
        return None

    temps = np.asarray(temps, dtype=float)
    heights = np.asarray(heights, dtype=float)
    mask = np.isfinite(temps) & np.isfinite(heights)
    if np.count_nonzero(mask) < 2:
        return None

    valid_indices = np.flatnonzero(mask)
    surface_idx = int(valid_indices[0])
    if surface_idx == trop_index:
        return None

    delta_temp = temps[trop_index] - temps[surface_idx]
    delta_height_km = (heights[trop_index] - heights[surface_idx]) / 1000.0
    if delta_height_km == 0:
        return None
    return -(delta_temp) / delta_height_km


def _height_at_pressure(pres: np.ndarray, heights: np.ndarray, target_pressure: float) -> float | None:
    """Interpolate geometric height for a given pressure level."""

    pres = np.asarray(pres, dtype=float)
    heights = np.asarray(heights, dtype=float)
    mask = np.isfinite(pres) & np.isfinite(heights)
    if np.count_nonzero(mask) < 2:
        return None

    pres_valid = pres[mask]
    heights_valid = heights[mask]
    order = np.argsort(pres_valid)
    pres_sorted = pres_valid[order]
    heights_sorted = heights_valid[order]

    if target_pressure < pres_sorted[0] or target_pressure > pres_sorted[-1]:
        return None

    return float(np.interp(target_pressure, pres_sorted, heights_sorted))


def _value_at_pressure(pres: np.ndarray, values: np.ndarray, target_pressure: float) -> float | None:
    """Interpolate an arbitrary sounding variable at a given pressure level."""

    pres = np.asarray(pres, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(pres) & np.isfinite(values)
    if np.count_nonzero(mask) < 2:
        return None

    pres_valid = pres[mask]
    values_valid = values[mask]
    order = np.argsort(pres_valid)
    pres_sorted = pres_valid[order]
    values_sorted = values_valid[order]

    if target_pressure < pres_sorted[0] or target_pressure > pres_sorted[-1]:
        return None

    return float(np.interp(target_pressure, pres_sorted, values_sorted))


def main() -> None:
    cfg = SoundingConfig()
    file_path = download_ua(cfg.region, cfg.station, cfg.year, cfg.month, cfg.day, cfg.hour)

    start_row, end_row = read_ua_html(file_path)
    data = importfile(file_path, start_row, end_row)

    pres = data["PRES"]
    hght = data["HGHT"]
    temp = data["TEMP"]
    dwpt = data["DWPT"]
    rhum = data["RELH"]
    mixr = data["MIXR"]
    drct = data["DRCT"]
    wspd = data["SKNT"] * 0.514
    evpr = (29.0 / 18.0) * pres * mixr / 1000.0

    figure_bg = "#eef6ff"
    axes_bg = "#fff5e6"

    fig1, ax1 = plt.subplots()
    fig1.patch.set_facecolor(figure_bg)
    ax1.set_facecolor(axes_bg)
    fig1.suptitle(cfg.plot_title, fontsize=14, fontweight="bold")
    ax1.plot(pres, hght / 1000.0, label="sounding")
    ax1.plot(1013.25 * np.power(10.0, (-hght / 17000.0)), hght / 1000.0, "--", label="standard atmosphere")
    ax1.set_xlabel("PRESSURE (hPa)")
    ax1.set_ylabel("HEIGHT (km)")
    ax1.grid(True)
    ax1.set_xlim(0, 1050)
    hght_500 = _height_at_pressure(pres, hght, 500.0)
    hght_1000 = _height_at_pressure(pres, hght, 1000.0)
    evpr_500 = _value_at_pressure(pres, evpr, 500.0)
    evpr_1000 = _value_at_pressure(pres, evpr, 1000.0)
    if hght_500 is not None:
        ax1.axhline(
            hght_500 / 1000.0,
            color="red",
            linestyle="--",
            linewidth=1.3,
            label="500 hPa level",
        )
    ax1.legend()
    ax1.tick_params(labelsize=12)

    gamma_p_01_02 = _safe_gradient(pres, hght, 6, 3)
    gamma_p_09_10 = _safe_gradient(pres, hght, 43, 37)

    fig2, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
    fig2.patch.set_facecolor(figure_bg)
    fig2.suptitle(cfg.plot_title, fontsize=14, fontweight="bold")

    axes[0, 0].plot(temp, hght / 1000.0, label="TEMP")
    axes[0, 0].plot(dwpt, hght / 1000.0, label="DWPT")
    axes[0, 0].grid(True)
    axes[0, 0].set_xlabel("TEMP/TDEW (°C)")
    axes[0, 0].set_ylabel("HEIGHT (km)")

    tropopause_idx = _find_tropopause_index(temp, hght)
    gamma_t_trop = _tropospheric_lapse_rate(temp, hght, tropopause_idx)
    if tropopause_idx is not None:
        axes[0, 0].axhline(
            hght[tropopause_idx] / 1000.0,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Tropopause height",
        )
    axes[0, 0].legend(loc="best", fontsize=9)

    gamma_t_00_02 = _safe_gradient(temp, hght, 10, 0)
    gamma_t_01_11 = _safe_gradient(temp, hght, 51, 0)

    axes[0, 1].plot(mixr, hght / 1000.0, label="MIXR")
    if len(mixr) > 1:
        axes[0, 1].plot(
            mixr[1] * np.power(10.0, (-hght / 7000.0)),
            hght / 1000.0,
            "--",
            label="standard atmosphere",
        )
    axes[0, 1].grid(True)
    axes[0, 1].set_xlabel("MIXR (g/kg)")
    axes[0, 1].legend(fontsize=9)

    mixr_sum = np.nansum(mixr)
    mixr_prc = np.zeros_like(mixr) if mixr_sum == 0 else mixr / mixr_sum
    wvapor00_05 = np.nansum(mixr_prc[:22])

    axes[0, 2].plot(evpr, hght / 1000.0, label="EVPR")
    if len(evpr) > 1:
        axes[0, 2].plot(
            evpr[1] * np.power(10.0, (-hght / 5000.0)),
            hght / 1000.0,
            "--",
            label="standard atmosphere",
        )
    axes[0, 2].grid(True)
    axes[0, 2].set_xlabel("EVPR (hPa)")
    evpr_xlim = axes[0, 2].get_xlim()
    x_label_pos = evpr_xlim[1] - 0.05 * (evpr_xlim[1] - evpr_xlim[0])
    if hght_1000 is not None:
        axes[0, 2].axhline(
            hght_1000 / 1000.0,
            color="tab:green",
            linestyle="--",
            linewidth=1.2,
            label="1000 hPa level",
        )
        if evpr_1000 is not None:
            axes[0, 2].text(
                x_label_pos,
                hght_1000 / 1000.0 + 0.1,
                f"1000 hPa: {evpr_1000:.1f} hPa",
                color="tab:green",
                fontsize=8,
                ha="right",
            )
    if hght_500 is not None:
        axes[0, 2].axhline(
            hght_500 / 1000.0,
            color="tab:purple",
            linestyle="--",
            linewidth=1.2,
            label="500 hPa level",
        )
        if evpr_500 is not None:
            axes[0, 2].text(
                x_label_pos,
                hght_500 / 1000.0 + 0.1,
                f"500 hPa: {evpr_500:.1f} hPa",
                color="tab:purple",
                fontsize=8,
                ha="right",
            )
    axes[0, 2].legend(fontsize=9)

    axes[1, 0].plot(rhum, hght / 1000.0)
    axes[1, 0].grid(True)
    axes[1, 0].set_xlabel("RHUM (%)")
    axes[1, 0].set_xlim(0, 100)

    axes[1, 1].plot(wspd, hght / 1000.0, label="WSPD")
    try:
        fit_wspd, fit_height = _polyfit_subset(hght, wspd, 20000.0)
        axes[1, 1].plot(fit_wspd, fit_height / 1000.0, label="polyfit", linestyle="--")
    except (ValueError, np.linalg.LinAlgError):
        pass
    axes[1, 1].grid(True)
    axes[1, 1].set_xlabel("WSPD (m/s)")
    axes[1, 1].legend(fontsize=9)

    axes[1, 2].plot(drct, hght / 1000.0)
    axes[1, 2].grid(True)
    axes[1, 2].set_xlabel("WDIR (deg)")
    axes[1, 2].set_xlim(0, 360)
    axes[1, 2].set_xticks(np.arange(0, 361, 90))

    for ax in axes.flat:
        ax.set_ylabel("HEIGHT (km)")
        ax.set_facecolor(axes_bg)

    lw = np.nansum(0.5 * (mixr[:-1] + mixr[1:]) * (-np.diff(pres))) / 100000.0
    thickness_1000_500 = None
    if hght_500 is not None and hght_1000 is not None:
        thickness_1000_500 = hght_500 - hght_1000

    print(f"Precipitable water = {1000.0 * lw:10.1f} mm")
    print(f"gammaP01_02 = {gamma_p_01_02:.4f}, gammaP09_10 = {gamma_p_09_10:.4f}")
    print(f"gammaT00_02 = {gamma_t_00_02:.4f}, gammaT01_11 = {gamma_t_01_11:.4f}")
    print(f"wvapor00_05 = {wvapor00_05:.4f}")
    if gamma_t_trop is not None:
        print(f"gammaT_trop = {gamma_t_trop:.4f} K/km")
    else:
        print("gammaT_trop = (not available)")
    if thickness_1000_500 is not None:
        print(f"Thickness_1000_500 = {thickness_1000_500/1000.0:.2f} km ({thickness_1000_500:.0f} m)")
    else:
        print("Thickness_1000_500 = (not available)")
    if evpr_1000 is not None:
        print(f"EVPR_1000hPa = {evpr_1000:.2f} hPa")
    else:
        print("EVPR_1000hPa = (not available)")
    if evpr_500 is not None:
        print(f"EVPR_500hPa = {evpr_500:.2f} hPa")
    else:
        print("EVPR_500hPa = (not available)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
