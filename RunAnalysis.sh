#!/bin/bash

# 1. FAIL-SAFE: Stop immediately if any command fails
set -e

echo "[1/4] Setting up Environment (LCG_105_cuda)..."
source /cvmfs/sft.cern.ch/lcg/views/LCG_105_cuda/x86_64-el9-gcc11-opt/setup.sh

# Ensure output directory exists before running
mkdir -p outputs

echo "[2/4] Running Simulation..."
# OPTIMIZATION: The '+' triggers ACLiC compilation.
# This compiles the C++ macro to a shared lib, speeding up the event loop significantly.
root -b -q "run_simulation.cc"

echo "[3/4] Running Neural Likelihood Estimation..."
# OPTIMIZATION: '-u' prevents python from buffering output (lets you see progress bars).
# 'tee' saves the log to a file for later inspection.
python neural_likelihood_ratio_estimation.py

echo "[4/4] Plotting Results..."
python plot_results.py

echo "Done! Results are in ./outputs/"