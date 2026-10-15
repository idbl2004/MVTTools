"""
Reproduce Fig. 1 of Dichiara et al. (2023, ApJL 954, L29).

Fermi/GBM light curves of GRB 230307A (bn230307656) in two energy bands:
    soft  7-99 keV
    hard  883-38367 keV
from the detectors used in the paper (NaI n6, n8, na and BGO b1).
The soft band uses the three NaI only, since BGO has no channels below
~150 keV. The hard band sums all four: this reproduces the published
counts, while b1 alone falls ~20% short.
16 ms bins, raw counts per bin (no background subtraction), with the
pile-up bad time interval marked and an inset on the first 2.5 s.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gdt.missions.fermi.gbm.tte import GbmTte

# --- Analysis settings, all taken from Dichiara et al. (2023) -------------
DATA_DIR = Path("data/fermiGBM/GRB230307A/current")
BURST_ID = "bn230307656"
BIN_WIDTH = 0.016                     # s
SOFT_BAND = (7.0, 99.0)               # keV
HARD_BAND = (883.0, 38367.0)          # keV
SOFT_DETECTORS = ["n6", "n8", "na"]         # BGO has no channels below ~150 keV
HARD_DETECTORS = ["n6", "n8", "na", "b1"]   # NaI counts above 883 keV are needed to match the paper
BAD_TIME_INTERVAL = (3.0, 7.0)        # s after trigger, pile-up
PLOT_RANGE = (-5.0, 50.0)             # s, x-axis of Fig. 1
INSET_RANGE = (-0.3, 2.6)             # s, first ~2.5 s


def band_lightcurve(detectors, energy_band, time_range):
    """Counts per 16 ms bin in one energy band, summed over detectors.

    Same approach as Derek's _combineDetectorData: keep only the photons in
    the energy band, then histogram their arrival times on one fixed grid so
    every detector shares exactly the same bins.
    """
    bin_edges = np.arange(time_range[0], time_range[1] + BIN_WIDTH, BIN_WIDTH)
    total_counts = np.zeros(len(bin_edges) - 1)
    for det in detectors:
        tte = GbmTte.open(DATA_DIR / f"glg_tte_{det}_{BURST_ID}_v00.fit")
        photons_in_band = tte.slice_energy(energy_band)
        arrival_times = np.asarray(photons_in_band.data.times)
        tte.close()  # TTE files are large; free the memory before the next one
        counts, _ = np.histogram(arrival_times, bins=bin_edges)
        total_counts += counts
    return bin_edges, total_counts


def draw_bands(ax, t_soft, soft, t_hard, hard, with_errors=True):
    """Hard band as a grey filled curve, soft band in red on top."""
    ax.fill_between(t_hard, hard, step="mid", color="lightgrey", zorder=1)
    ax.step(t_hard, hard, where="mid", color="grey", lw=0.6, zorder=2)
    if with_errors:
        # Poisson uncertainty on N counts is sqrt(N)
        ax.errorbar(t_soft, soft, yerr=np.sqrt(soft), fmt="none",
                    ecolor="red", elinewidth=0.5, zorder=3)
    ax.step(t_soft, soft, where="mid", color="red", lw=0.8, zorder=4)


# --- Build the two light curves -------------------------------------------
soft_edges, soft_counts = band_lightcurve(SOFT_DETECTORS, SOFT_BAND, PLOT_RANGE)
hard_edges, hard_counts = band_lightcurve(HARD_DETECTORS, HARD_BAND, PLOT_RANGE)
t_soft = 0.5 * (soft_edges[:-1] + soft_edges[1:])
t_hard = 0.5 * (hard_edges[:-1] + hard_edges[1:])

# Quick numbers to compare by eye with the published figure
print(f"soft band, pre-burst level: {soft_counts[t_soft < -1.0].mean():.0f} cts/bin")
print(f"hard band, pre-burst level: {hard_counts[t_hard < -1.0].mean():.0f} cts/bin")
print(f"soft band, peak: {soft_counts.max():.0f} cts/bin at "
      f"t = {t_soft[soft_counts.argmax()]:.3f} s")
print(f"hard band, peak: {hard_counts.max():.0f} cts/bin at "
      f"t = {t_hard[hard_counts.argmax()]:.3f} s")

# --- Main figure ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))
draw_bands(ax, t_soft, soft_counts, t_hard, hard_counts)

ax.set_xlim(*PLOT_RANGE)
ax.set_ylim(bottom=-15)
ax.set_xlabel("Time since GBM trigger [s]")
ax.set_ylabel("Counts [cts/bin]")

# Pile-up interval: dashed line along the bottom, as in the paper
ax.hlines(0, *BAD_TIME_INTERVAL, colors="grey", linestyles="--", lw=1.5)
ax.text(np.mean(BAD_TIME_INTERVAL), 8, "BTI", color="grey", ha="center")
# Thick bar marking the zoomed interval
ax.hlines(0, *INSET_RANGE, colors="black", lw=4)

ax.plot([], [], color="red", label=f"{SOFT_BAND[0]:.0f}-{SOFT_BAND[1]:.0f} keV")
ax.plot([], [], color="grey", label=f"{HARD_BAND[0]:.0f}-{HARD_BAND[1]:.0f} keV")
ax.legend(loc="upper right")

# --- Inset: first 2.5 s ---------------------------------------------------
inset = ax.inset_axes([0.6, 0.38, 0.35, 0.4])
zoom_soft = (t_soft > INSET_RANGE[0]) & (t_soft < INSET_RANGE[1])
zoom_hard = (t_hard > INSET_RANGE[0]) & (t_hard < INSET_RANGE[1])
draw_bands(inset, t_soft[zoom_soft], soft_counts[zoom_soft],
           t_hard[zoom_hard], hard_counts[zoom_hard], with_errors=False)
inset.set_xlim(*INSET_RANGE)
inset.set_xlabel("Time [s]")
precursor = (t_soft > INSET_RANGE[0]) & (t_soft < 0.3)
i_peak = soft_counts[precursor].argmax()
inset.annotate("candidate\nprecursor",
               xy=(t_soft[precursor][i_peak], soft_counts[precursor][i_peak]),
               xytext=(0.3, 0.8), textcoords="axes fraction",
               arrowprops=dict(arrowstyle="->"), fontsize=11)

fig.tight_layout()
fig.savefig("ignacio/dichiara_fig1_reproduced.png", dpi=200)
print("saved ignacio/dichiara_fig1_reproduced.png")
