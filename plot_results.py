import numpy as np
import matplotlib.pyplot as plt
import mplhep as mh
import uproot
import awkward as ak
import ROOT

# Plotting Style
from matplotlib import rc
rc('font', family='serif')
rc('font', size=15)
rc('legend', fontsize=15)
rc('lines', linewidth=2)

# ==========================================
# 1. Load Data
# ==========================================

# Load Histograms (uproot reads them as specific objects mplhep understands)
with uproot.open("./outputs/simulation.root") as simulation:
    HistSpinUp_X1 = simulation["HistSpinUp_X1"]        # Data (AN=0.2)
    HistSpinDown_X1 = simulation["HistSpinDown_X1"]
    HistSpinUp_X0_test = simulation["HistSpinUp_X0_test"]  # Sim (AN=0.0)
    HistSpinDown_X0_test = simulation["HistSpinDown_X0_test"]

# Load Arrays
with uproot.open("./outputs/result.root") as result:
    AN_runs = result["result"]["AN"].array(library="np")      # Stat Bootstrap
    model_runs = result["model"]["AN"].array(library="np")    # Systematics
    counts_spin_up = result["counts_spin_up"]["counts"].array(library="np")
    counts_spin_down = result["counts_spin_down"]["counts"].array(library="np")

# Helper for text box
def add_stat_box(ax, data, label=""):
    mean = np.mean(data)
    std = np.std(data)
    text_str = f"Mean: {mean:.4f}\n" \
               f"Std:  {std:.4f}"
    # Place text in top right
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ==========================================
# 2. Physics Plots (Phi Distributions)
# ==========================================
pi = np.pi
n_bins_phi = np.linspace(-pi, pi, 13)

# --- Spin Up ---
fig, ax = plt.subplots(figsize=(6, 6))
mh.histplot(HistSpinUp_X1, histtype="fill", label=r"Target ($A_{N} = 0.2$)", color="forestgreen", alpha=0.3, density=True)
mh.histplot(HistSpinUp_X0_test, histtype="step", label=r"Source ($A_{N} = 0.0$)", color="dodgerblue", linewidth=2, density=True)
mh.histplot(np.mean(counts_spin_up, axis=0), bins=n_bins_phi, yerr=np.std(counts_spin_up, axis=0), histtype="step", label=r"Source Reweighted", color="forestgreen", linewidth=2, linestyle="--", density=True)
ax.set_xlabel(r"$\phi$ [rad]")
ax.set_ylabel("Normalized to unity")
plt.legend(frameon=False, loc="lower center") # Moved legend to avoid overlap
plt.tight_layout()
plt.savefig("./outputs/SpinUp.png")
plt.close()

# --- Spin Down ---
fig, ax = plt.subplots(figsize=(6, 6))
mh.histplot(HistSpinDown_X1, histtype="fill", label=r"Target ($A_{N} = 0.2$)", color="forestgreen", alpha=0.3, density=True)
mh.histplot(HistSpinDown_X0_test, histtype="step", label=r"Source ($A_{N} = 0.0$)", color="dodgerblue", linewidth=2, density=True)
mh.histplot(np.mean(counts_spin_down, axis=0), bins=n_bins_phi, yerr=np.std(counts_spin_down, axis=0), histtype="step", label=r"Source Reweighted", color="forestgreen", linewidth=2, linestyle="--", density=True)

ax.set_xlabel(r"$\phi$ [rad]")
ax.set_ylabel("Normalized to unity")
plt.legend(frameon=False, loc="lower center")
plt.tight_layout()
plt.savefig("./outputs/SpinDown.png")
plt.close()

# ==========================================
# 3. Analysis Plots (AN Distributions)
# ==========================================

# Define bins for AN plots
n_bins_an = np.linspace(0.15, 0.25, 21) # Adjusted bins for clarity

# --- Statistical Uncertainty (Bootstrap) ---
fig, ax = plt.subplots(figsize=(6, 6))

counts, edges = np.histogram(AN_runs, bins=n_bins_an)
mh.histplot(counts, bins=edges, histtype="fill", color="dodgerblue", alpha=0.3, linewidth=2, edgecolor='blue')

ax.set_xlabel(r"Extracted $A_{N}$")
ax.set_ylabel("Bootstrap Iterations")
ax.set_title("Statistical Uncertainty")

# Add Mean/Error Text
add_stat_box(ax, AN_runs, label="Stat. Bootstrap")

plt.tight_layout()
plt.savefig("./outputs/ANRuns.png")
plt.close()

# --- Systematic Uncertainty (Model) ---
fig, ax = plt.subplots(figsize=(6, 6))

counts, edges = np.histogram(model_runs, bins=n_bins_an)
mh.histplot(counts, bins=edges, histtype="fill", color="crimson", alpha=0.3, linewidth=2, edgecolor='red')

ax.set_xlabel(r"Extracted $A_{N}$")
ax.set_ylabel("Training Runs")
ax.set_title("Model Uncertainty")

# Add Mean/Error Text
add_stat_box(ax, model_runs, label="Model Systematics")

plt.tight_layout()
plt.savefig("./outputs/ModelRuns.png")
plt.close()