"""
Background-subtracted light curves of GRB 230307A (bn230307656).

Improvement 1 on top of the reproduction of Dichiara et al. (2023) Fig. 1.
Fig. 1 shows raw counts, so the background (~42 cts/bin soft, ~11 hard)
is still in it. Here the background is modelled and removed, using the
same recipe Dichiara use for their spectra: a 3rd-order polynomial in time
fitted to -89..-7 s and +162..+273 s, channel by channel.

Outputs
-------
background_fit_check.png
    Raw counts in 1.024 s bins over the whole fit range with the model on
    top and the fit windows shaded: shows whether the polynomial is a fair
    description of the quiet periods, and how it interpolates under the burst.
lightcurves_bkg_subtracted.png
    Net counts per 16 ms bin, same bands and detectors as Fig. 1.
    Error on each bin = sqrt(N + sigma_bkg^2): Poisson noise on the raw
    counts plus the uncertainty of the background model.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lightcurve_tools import (aligned_bin_edges, band_counts, fit_background,
                              background_in_band)

# --- Settings (same as the Fig. 1 reproduction) ---------------------------
DATA_DIR = "data/fermiGBM/GRB230307A/current"
BURST_ID = "bn230307656"
BANDS = {
    "7-99 keV": {"range": (7.0, 99.0), "detectors": ["n6", "n8", "na"],
                 "colour": "red"},
    "883-38367 keV": {"range": (883.0, 38367.0),
                      "detectors": ["n6", "n8", "na", "b1"], "colour": "grey"},
}
BKG_WINDOWS = [(-89.0, -7.0), (162.0, 273.0)]   # s, from Dichiara Sec. 2
BKG_ORDER = 3
FINE_BIN = 0.016        # s
COARSE_BIN = 1.024      # s, for the fit check plot
PLOT_RANGE = (-5.0, 50.0)
FULL_RANGE = (-100.0, 280.0)
BAD_TIME_INTERVAL = (3.0, 7.0)

fine_edges = aligned_bin_edges(PLOT_RANGE, FINE_BIN)
coarse_edges = aligned_bin_edges(FULL_RANGE, COARSE_BIN)
fine_t = 0.5 * (fine_edges[:-1] + fine_edges[1:])
coarse_t = 0.5 * (coarse_edges[:-1] + coarse_edges[1:])

# One background fit per detector, reused for every band and binning
fitters = {}
for det in sorted({d for band in BANDS.values() for d in band["detectors"]}):
    fitters[det], chi2 = fit_background(DATA_DIR, BURST_ID, det, BKG_WINDOWS, BKG_ORDER)
    print(f"{det}: background fit reduced chi2 = {chi2:.2f}")

results = {}
for name, band in BANDS.items():
    raw = np.zeros(len(fine_t))
    bkg = np.zeros(len(fine_t))
    bkg_var = np.zeros(len(fine_t))
    raw_coarse = np.zeros(len(coarse_t))
    bkg_coarse = np.zeros(len(coarse_t))
    for det in band["detectors"]:
        raw += band_counts(DATA_DIR, BURST_ID, det, band["range"], fine_edges)
        b, b_err = background_in_band(fitters[det], band["range"], fine_edges)
        bkg += b
        bkg_var += b_err**2   # detectors are fitted independently
        raw_coarse += band_counts(DATA_DIR, BURST_ID, det, band["range"], coarse_edges)
        bkg_coarse += background_in_band(fitters[det], band["range"], coarse_edges)[0]

    net = raw - bkg
    net_err = np.sqrt(raw + bkg_var)
    results[name] = dict(raw=raw, bkg=bkg, net=net, net_err=net_err,
                         raw_coarse=raw_coarse, bkg_coarse=bkg_coarse)

    at_trigger = np.argmin(np.abs(fine_t))
    pre = fine_t < -1.0
    print(f"{name}: background at T0 = {bkg[at_trigger]:.1f} "
          f"+/- {np.sqrt(bkg_var[at_trigger]):.1f} cts/bin; "
          f"net counts before -1 s average {net[pre].mean():.1f} (should be ~0)")

# --- Check of the background fit -------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for ax, (name, band) in zip(axes, BANDS.items()):
    r = results[name]
    ax.step(coarse_t, r["raw_coarse"], where="mid", color=band["colour"], lw=0.8,
            label=f"{name} raw")
    ax.plot(coarse_t, r["bkg_coarse"], color="black", lw=1.2,
            label=f"poly{BKG_ORDER} background")
    for lo, hi in BKG_WINDOWS:
        ax.axvspan(lo, hi, color="tab:blue", alpha=0.1)
    # Zoom the y-axis on the background level; the burst goes off the top
    quiet = r["bkg_coarse"]
    ax.set_ylim(0.8 * quiet.min(), 1.6 * quiet.max())
    ax.set_ylabel(f"Counts per {COARSE_BIN} s")
    ax.legend(loc="upper right")
axes[-1].set_xlabel("Time since GBM trigger [s]")
axes[0].set_title("Shaded: background fit windows")
fig.tight_layout()
fig.savefig("ignacio/background_fit_check.png", dpi=200)

# --- Background-subtracted light curves ------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for ax, (name, band) in zip(axes, BANDS.items()):
    r = results[name]
    ax.errorbar(fine_t, r["net"], yerr=r["net_err"], fmt="none",
                ecolor=band["colour"], elinewidth=0.4, alpha=0.4)
    ax.plot(fine_t, r["net"], color=band["colour"], lw=0.7, label=name)
    ax.axhline(0, color="black", lw=0.6)
    ax.axvspan(*BAD_TIME_INTERVAL, color="grey", alpha=0.15, label="BTI (pile-up)")
    ax.set_ylabel("Net counts [cts/bin]")
    ax.legend(loc="upper right")
axes[-1].set_xlim(*PLOT_RANGE)
axes[-1].set_xlabel("Time since GBM trigger [s]")
fig.tight_layout()
fig.savefig("ignacio/lightcurves_bkg_subtracted.png", dpi=200)
print("saved ignacio/background_fit_check.png and ignacio/lightcurves_bkg_subtracted.png")