import numpy as np
import os
import uproot
import awkward as ak
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.utils import resample
from OmniFold import MLPClassifier, MLPDataset, train_mlp, calculate_likelihood_ratio

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ==========================================
# 1. Physics & Helper Functions
# ==========================================

def log_likelihood(weights, P, phi_pol, x):
    # Ensure inputs are numpy arrays for proper broadcasting
    weights = np.array(weights)
    P = np.array(P)
    phi_pol = np.array(phi_pol)
    x = np.array(x)
    
    numerator = weights * P * np.sin(phi_pol - x)
    denominator = weights * (P**2) * (np.sin(phi_pol - x)**2)
    
    return np.sum(numerator) / np.sum(denominator)

def fill_histogram(xvals, weights, n_bins:int = 12):
    n_edges = np.linspace(-np.pi, np.pi, n_bins+1)
    counts, _ = np.histogram(xvals, bins=n_edges, weights=weights)
    return counts

def run_training_pipeline(X0, W0, X1, W1, X_test, W_test, config):
    """
    Helper to train one MLP and return inference results.
    """
    # 1. Dataset Prep
    # Concatenate Simulation (Class 0) and Data (Class 1)
    X_train = np.concatenate((X0, X1))
    W_train = np.concatenate((W0, W1))
    Y_train = np.concatenate((np.zeros(len(X0)), np.ones(len(X1))))

    ds = MLPDataset(X_train, Y_train, W_train)
    
    train_len = int(0.5 * len(ds))
    val_len = len(ds) - train_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)

    # 2. Train
    mlp = MLPClassifier(x_dim=1).to(config['device'])
    mlp, _, _ = train_mlp(mlp, train_loader, val_loader, 
                          config['lr'], config['epochs'], config['device'])

    # 3. Inference
    # Dummy labels for test set
    Y_test_dummy = np.zeros(len(X_test))
    test_ds = MLPDataset(X_test, Y_test_dummy, W_test)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)
    
    x_out, w_out = calculate_likelihood_ratio(mlp, test_loader, config['device'])
    return x_out, w_out

# ==========================================
# 2. Data Loading
# ==========================================
print("Loading data from ROOT file...")

with uproot.open("./outputs/simulation.root") as simulation:
    # --- Simulation (X0) ---
    X0_spin_up = simulation["TreeSpinUp_X0"]["X"].array(library="np")
    W0_spin_up = simulation["TreeSpinUp_X0"]["W"].array(library="np")
    
    X0_spin_down = simulation["TreeSpinDown_X0"]["X"].array(library="np")
    W0_spin_down = simulation["TreeSpinDown_X0"]["W"].array(library="np")

    # --- Data (X1) ---
    X1_spin_up = simulation["TreeSpinUp_X1"]["X"].array(library="np")
    W1_spin_up = simulation["TreeSpinUp_X1"]["W"].array(library="np")

    X1_spin_down = simulation["TreeSpinDown_X1"]["X"].array(library="np")
    W1_spin_down = simulation["TreeSpinDown_X1"]["W"].array(library="np")

    # --- Test Set (X0 Test) ---
    X0_test_spin_up = simulation["TreeSpinUp_X0_test"]["X"].array(library="np")
    W0_test_spin_up = simulation["TreeSpinUp_X0_test"]["W"].array(library="np")

    X0_test_spin_down = simulation["TreeSpinDown_X0_test"]["X"].array(library="np")
    W0_test_spin_down = simulation["TreeSpinDown_X0_test"]["W"].array(library="np")

# ==========================================
# 3. Analysis Configuration
# ==========================================
config = {
    'batch_size': 2048,
    'lr': 1e-04,
    'epochs': 30,
    'device': device
}

n_runs = 100
P_vals = [0.9, 0.7]

# Containers
AN_stat = [] # Statistical Uncertainty (Bootstrap Data)
AN_sys = []  # Model Uncertainty (Retrain same data)
all_counts_up = []
all_counts_down = []

# ==========================================
# 4. Loop 1: Statistical Uncertainty (Bootstrap)
# ==========================================
print("\n[ === Starting Statistical Bootstrap Loop === ]")
for run in range(n_runs):
    print(f"--- Stat Run {run+1}/{n_runs} ---")
    
    # A. Resample DATA (X1) for bootstrap
    X1b_up, W1b_up = resample(X1_spin_up, W1_spin_up, replace=True)
    X1b_down, W1b_down = resample(X1_spin_down, W1_spin_down, replace=True)

    # B. Train Spin Up
    x_test_up, w_nn_up = run_training_pipeline(
        X0_spin_up, W0_spin_up, X1b_up, W1b_up, 
        X0_test_spin_up, W0_test_spin_up, config
    )
    all_counts_up.append(fill_histogram(x_test_up, w_nn_up))

    # C. Train Spin Down
    x_test_down, w_nn_down = run_training_pipeline(
        X0_spin_down, W0_spin_down, X1b_down, W1b_down, 
        X0_test_spin_down, W0_test_spin_down, config
    )
    all_counts_down.append(fill_histogram(x_test_down, w_nn_down))

    # D. Extract AN
    x_full = np.concatenate((x_test_up, x_test_down))
    w_full = np.concatenate((w_nn_up, w_nn_down))
    
    P_full = np.concatenate((P_vals[0] * np.ones(len(x_test_up)), P_vals[1] * np.ones(len(x_test_down))))
    phi_full = np.concatenate((np.pi/2 * np.ones(len(x_test_up)), -np.pi/2 * np.ones(len(x_test_down))))

    val = log_likelihood(w_full, P_full, phi_full, x_full)
    AN_stat.append(val)
    print(f"   -> Stat AN: {val:.4f}")

# ==========================================
# 5. Loop 2: Systematic Uncertainty (Model Stability)
# ==========================================
print("\n[ === Starting Model Systematic Loop === ]")
for run in range(n_runs):
    print(f"--- Sys Run {run+1}/{n_runs} ---")
    
    # DO NOT RESAMPLE DATA. Use original X1.
    
    # A. Train Spin Up
    x_test_up, w_nn_up = run_training_pipeline(
        X0_spin_up, W0_spin_up, X1_spin_up, W1_spin_up, 
        X0_test_spin_up, W0_test_spin_up, config
    )

    # B. Train Spin Down
    x_test_down, w_nn_down = run_training_pipeline(
        X0_spin_down, W0_spin_down, X1_spin_down, W1_spin_down, 
        X0_test_spin_down, W0_test_spin_down, config
    )

    # C. Extract AN
    x_full = np.concatenate((x_test_up, x_test_down))
    w_full = np.concatenate((w_nn_up, w_nn_down))
    
    P_full = np.concatenate((P_vals[0] * np.ones(len(x_test_up)), P_vals[1] * np.ones(len(x_test_down))))
    phi_full = np.concatenate((np.pi/2 * np.ones(len(x_test_up)), -np.pi/2 * np.ones(len(x_test_down))))

    val = log_likelihood(w_full, P_full, phi_full, x_full)
    AN_sys.append(val)
    print(f"   -> Sys AN: {val:.4f}")

# ==========================================
# 6. Final Results
# ==========================================
AN_mean = np.mean(AN_stat) # Central value comes from stat loop mean (or central fit)
AN_err_stat = np.std(AN_stat) # Spread of bootstrap
AN_err_sys = np.std(AN_sys)   # Spread of model retraining

print("-" * 40)
print(f"FINAL RESULT: AN = {AN_mean:.4f} +/- {AN_err_stat:.4f} (stat) +/- {AN_err_sys:.4f} (sys)")
print("-" * 40)

# Save
with uproot.recreate("./outputs/result.root") as result:
    result["result"] = {"AN": np.array(AN_stat)}
    result["model"] = {"AN": np.array(AN_sys)}
    result["counts_spin_up"] = {"counts": np.array(all_counts_up)}
    result["counts_spin_down"] = {"counts": np.array(all_counts_down)}

print("Saved to ./outputs/result.root")
