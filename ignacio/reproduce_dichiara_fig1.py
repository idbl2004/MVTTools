"""
Reproduce Fig. 1 of Dichiara et al. (2023, ApJL 954, L29).

Fermi/GBM light curves of GRB 230307A (bn230307656) in two energy bands:
    soft  7-99 keV
    hard  883-38367 keV
16 ms bins, raw counts per bin (no background subtraction), with the
pile-up bad time interval (3-7 s) marked and an inset on the first 2.5 s.

Detectors
---------
The paper uses NaI n6, n8, na and BGO b1, but does not say which of them
go into each band.
- Soft band: NaI only. BGO channels start at ~150 keV, so b1 has nothing
  in 7-99 keV (GDT raises an error if you ask it to).
- Hard band: all four. The edges 883 and 38367 keV are b1 channel
  boundaries (channels 20-123), so the band is defined from b1. On its
  own b1 peaks at ~450 cts/bin at 1.83 s, well below the ~600 in the
  paper. Applying the same energy range to all four detectors, keeping
  every channel that overlaps it (as slice_energy does), gives 579 with
  ~11 cts/bin before the burst, matching the paper (~600 and ~10). b0+b1
  overshoots (~700, ~17). Most of the NaI contribution comes from their
  top channel, which runs from ~1 MeV up to 2000 keV.
  So "all four detectors, one energy range" is my inference from the
  counts (check_hard_band.py), not something the paper states.

Counts vs flux
--------------
The y-axis is detected photons per bin summed over the chosen detectors,
not the photon flux from the burst. More detectors, or more efficient
ones, give more counts for the same source. Raw counts therefore depend
on the detector choice and can't be compared across detectors or papers
directly; the spectral fits use each detector's response (.rsp2) to get
back to the real flux.

Check against the paper
-----------------------
Printed values (see below) compared by eye with the published figure
(bins aligned to T0):
    soft  pre-burst 42 cts/bin, peak 604 at 6.104 s
    hard  pre-burst 11 cts/bin, peak 579 at 1.832 s
Precursor peaks in the inset: ~520 and ~450 cts/bin (paper ~500 and ~450).
The peak heights can be compared by eye; the pre-burst levels differ by
a few counts and can't. A digitised version of the paper's figure would
make this comparison quantitative.
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
INSET_RANGE = (-0.2, 2.6)             # s, data shown in the inset
# Layout measured off the published figure (pixel positions of its ticks)
ZOOM_BAR = (-0.2, 1.2)                # s, thick black bar; shorter than the
                                      # 2.5 s the caption mentions
INSET_POSITION = [0.65, 0.47, 0.32, 0.32]   # inset box, fraction of main axes
FILL_FLOOR = -100                     # bottom of the grey fill, cts/bin


def band_lightcurve(detectors, energy_band, time_range):
    """Counts per 16 ms bin in one energy band, summed over detectors.

    Same approach as Derek's _combineDetectorData: keep only the photons in
    the energy band, then histogram their arrival times on one fixed grid so
    every detector shares exactly the same bins.
    """
    # Build the edges from whole bin numbers so that one edge sits exactly on
    # the trigger (t = 0); starting at -5 s would put every edge 8 ms off
    first_bin = np.floor(time_range[0] / BIN_WIDTH)
    last_bin = np.ceil(time_range[1] / BIN_WIDTH)
    bin_edges = np.arange(first_bin, last_bin + 1) * BIN_WIDTH
    total_counts = np.zeros(len(bin_edges) - 1)

    print(f"{energy_band[0]:g}-{energy_band[1]:g} keV requested, channels used:")
    for det in detectors:
        tte = GbmTte.open(DATA_DIR / f"glg_tte_{det}_{BURST_ID}_v00.fit")
        photons_in_band = tte.slice_energy(energy_band)
        arrival_times = np.asarray(photons_in_band.data.times)

        # slice_energy keeps whole channels, so the real band runs from the
        # bottom of the lowest channel kept to the top of the highest one
        channels = np.asarray(photons_in_band.data.channels)
        low_edges = np.asarray(tte.ebounds.low_edges())
        high_edges = np.asarray(tte.ebounds.high_edges())
        lo_chan, hi_chan = int(channels.min()), int(channels.max())
        print(f"  {det}: channels {lo_chan}-{hi_chan} = "
              f"{low_edges[lo_chan]:.1f}-{high_edges[hi_chan]:.1f} keV "
              f"(detector's top channel ends at {high_edges[-1]:.1f} keV)")

        tte.close()  # TTE files are large; free the memory before the next one
        counts, _ = np.histogram(arrival_times, bins=bin_edges)
        total_counts += counts
    return bin_edges, total_counts


def draw_bands(ax, t_soft, soft, t_hard, hard, with_errors=True):
    """Hard band as a grey filled curve, soft band in red on top.

    Lines through the bin centres, like the paper. The main panel has
    sqrt(N) error bars on the soft band; the inset in the paper has none.
    """
    ax.fill_between(t_hard, hard, y2=FILL_FLOOR, color="lightgrey", zorder=1)
    ax.plot(t_hard, hard, color="darkgrey", lw=0.8, zorder=2)
    if with_errors:
        # Poisson uncertainty on N counts is sqrt(N)
        ax.errorbar(t_soft, soft, yerr=np.sqrt(soft), fmt="none",
                    ecolor="red", elinewidth=0.6, capsize=1.5, zorder=3)
        ax.plot(t_soft, soft, color="red", lw=0.8, zorder=4)
    else:
        ax.plot(t_soft, soft, color="red", lw=1.5, zorder=4)


# --- Build the two light curves -------------------------------------------
soft_edges, soft_counts = band_lightcurve(SOFT_DETECTORS, SOFT_BAND, PLOT_RANGE)
hard_edges, hard_counts = band_lightcurve(HARD_DETECTORS, HARD_BAND, PLOT_RANGE)
t_soft = 0.5 * (soft_edges[:-1] + soft_edges[1:])
t_hard = 0.5 * (hard_edges[:-1] + hard_edges[1:])

# Numbers to compare with the published figure (values in the docstring)
print(f"soft band, pre-burst level: {soft_counts[t_soft < -1.0].mean():.0f} cts/bin")
print(f"hard band, pre-burst level: {hard_counts[t_hard < -1.0].mean():.0f} cts/bin")
print(f"soft band, peak: {soft_counts.max():.0f} cts/bin at "
      f"t = {t_soft[soft_counts.argmax()]:.3f} s")
print(f"hard band, peak: {hard_counts.max():.0f} cts/bin at "
      f"t = {t_hard[hard_counts.argmax()]:.3f} s")

# --- Main figure ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6.5))
draw_bands(ax, t_soft, soft_counts, t_hard, hard_counts)

ax.set_xlim(*PLOT_RANGE)
ax.set_ylim(bottom=-10)
ax.set_xlabel("Time since GBM trigger [s]")
ax.set_ylabel("Counts [cts/bin]")

# Pile-up interval: dashed line along the bottom, as in the paper
ax.hlines(0, *BAD_TIME_INTERVAL, colors="grey", linestyles="--", lw=1.5)
ax.text(np.mean(BAD_TIME_INTERVAL), 8, "BTI", color="grey", ha="center")
# Thick bar marking the start of the burst
ax.hlines(0, *ZOOM_BAR, colors="black", lw=4)

ax.plot([], [], color="red", label=f"{SOFT_BAND[0]:.0f}-{SOFT_BAND[1]:.0f} keV")
ax.plot([], [], color="darkgrey", lw=0.8, label=f"{HARD_BAND[0]:.0f}-{HARD_BAND[1]:.0f} keV")
ax.legend(loc="upper right")

# --- Inset: first 2.5 s ---------------------------------------------------
inset = ax.inset_axes(INSET_POSITION)
zoom_soft = (t_soft > INSET_RANGE[0]) & (t_soft < INSET_RANGE[1])
zoom_hard = (t_hard > INSET_RANGE[0]) & (t_hard < INSET_RANGE[1])
draw_bands(inset, t_soft[zoom_soft], soft_counts[zoom_soft],
           t_hard[zoom_hard], hard_counts[zoom_hard], with_errors=False)
inset.set_xticks([0, 1, 2])   # axis limits left automatic, as in the paper
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
