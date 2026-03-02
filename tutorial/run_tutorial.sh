#!/bin/bash

echo "[1/5] Setting up Environment (LCG_105_cuda)..."
source /cvmfs/sft.cern.ch/lcg/views/LCG_105_cuda/x86_64-el9-gcc11-opt/setup.sh

# Ensure output directory exists before running
mkdir -p ./build ./outputs ./outputs/imgs

echo "[2/5] Compiling libraries ..."
g++ -O3 -march=native ./tutorial/run_simulation.cc ./source/TreeHelper.cc -o ./build/simulation -I ./include $(root-config --cflags --libs)

echo "[3/5] Running Simulation..."
time ./build/simulation

echo "[4/5] Running Neural Likelihood Estimation..."
python ./tutorial/neural_likelihood_ratio_estimation.py

echo "[5/5] Plotting Results..."
python ./tutorial/plot_results.py

echo "Done! Results are in ./outputs/imgs"
