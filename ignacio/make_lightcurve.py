# make_lightcurve.py: light curve of GRB 230307A from Fermi GBM TTE data (NaI n6)
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save plots to files, don't try to open a window
import matplotlib.pyplot as plt
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.core.binning.unbinned import bin_by_time

# Load the photon list for detector n6
tte = GbmTte.open("data/fermiGBM/GRB230307A/current/glg_tte_n6_bn230307656_v00.fit")

# Count photons in 16 ms boxes (as Dichiara et al. 2023), then sum 8-1000 keV
phaii = tte.to_phaii(bin_by_time, 0.016)
lc = phaii.to_lightcurve(energy_range=(8, 1000))

# Make two plots: the whole burst, and a zoom on the precursor
for name, (t0, t1) in {"full": (-10, 60), "precursor": (-0.2, 0.5)}.items():
    fig, ax = plt.subplots()
    ax.step(lc.centroids, lc.rates, where="mid")
    ax.set_xlim(t0, t1)
    inside = (lc.centroids > t0) & (lc.centroids < t1)
    ax.set_ylim(0, 1.1 * lc.rates[inside].max())  # scale y-axis to what's visible
    ax.set_xlabel("Time since trigger (s)")
    ax.set_ylabel("Count rate (counts/s)")
    ax.set_title(f"GRB 230307A, NaI n6, 8-1000 keV, 16 ms bins ({name})")
    if name == "precursor":
        ax.axvspan(-0.064, 0.320, alpha=0.2, label="Dichiara precursor window")
        ax.legend()
    fig.savefig(f"ignacio/lc_n6_{name}.png", dpi=150)
    plt.close(fig)
    print("saved", f"ignacio/lc_n6_{name}.png")
    