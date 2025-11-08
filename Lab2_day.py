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
    month: int = 2
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
    print(f"Precipitable water = {1000.0 * lw:10.1f} mm")
    print(f"gammaP01_02 = {gamma_p_01_02:.4f}, gammaP09_10 = {gamma_p_09_10:.4f}")
    print(f"gammaT00_02 = {gamma_t_00_02:.4f}, gammaT01_11 = {gamma_t_01_11:.4f}")
    print(f"wvapor00_05 = {wvapor00_05:.4f}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
