#include "TFile.h"
#include <iostream>
#include <memory>

#include "TreeHelper.hh"

int main() {

    // --- Train test events ---
    const int num_samples = 200000;

    // --- Beam configuration ---
    const double pol_down = 0.9;
    const double pol_up = 0.7;
    const double lumi_up = 0.6;
    const double lumi_down = 0.4;

    // --- Physics configuration ---
    const double AN_sim = 0.0; // Simulation (No Asymmetry)
    const double AN_data = 0.2; // Data (Target)

    // --- Weights calculation ---
    double weight_up = (lumi_up* pol_up + lumi_down* pol_down)/(2.0* lumi_up* pol_up);
    double weight_down = (lumi_up* pol_up + lumi_down* pol_down)/(2.0* lumi_down* pol_down);

    int num_spin_up = (int)(lumi_up* num_samples);
    int num_spin_down = num_samples - num_spin_up;

    std::cout << " > Spin up events " << num_spin_up << std::endl;
    std::cout << " > Spin down events " << num_spin_down << std::endl;

    auto fo = std::make_unique<TFile>("./outputs/simulation.root", "RECREATE");

    auto cross_section = std::make_unique<TF1>("cross_section", "(1.0/(2.0* TMath::Pi()))* (1.0 + [0]* [1]* TMath::Sin([2] - x))", -TMath::Pi(), TMath::Pi());

    // --- Simulated events ---
    // --- Spin up (label = 0) ---
    std::cout << "  > Simulated events ..." << std::endl;
    cross_section->SetParameters(pol_up, AN_sim, TMath::PiOver2());

    auto X0_spin_up_tree = std::make_unique<TreeHelper>("X0_spin_up_tree");

    X0_spin_up_tree->samples(num_spin_up, weight_up, 0.0, cross_section.get());
    X0_spin_up_tree->write();

    // --- Spin down (label = 0) ---
    cross_section->SetParameters(pol_down, AN_sim, -TMath::PiOver2());

    auto X0_spin_down_tree = std::make_unique<TreeHelper>("X0_spin_down_tree");

    X0_spin_down_tree->samples(num_spin_down, weight_down, 0.0, cross_section.get());
    X0_spin_down_tree->write();

    // --- Pseudo data (target) ---
    // --- Spin up (label = 1) ---
    std::cout << "  > Pseudo experiment events ..." << std::endl;
    cross_section->SetParameters(pol_up, AN_data, TMath::PiOver2());

    auto X1_spin_up_tree = std::make_unique<TreeHelper>("X1_spin_up_tree");

    X1_spin_up_tree->samples(num_spin_up, weight_up, 1.0, cross_section.get());
    X1_spin_up_tree->write();

    // --- Spin down ---
    cross_section->SetParameters(pol_down, AN_data, -TMath::PiOver2());

    auto X1_spin_down_tree = std::make_unique<TreeHelper>("X1_spin_down_tree");

    X1_spin_down_tree->samples(num_spin_down, weight_down, 1.0, cross_section.get());
    X1_spin_down_tree->write();

    // --- Test simulation ---
    // --- Spin up (label = 0) ---
    std::cout << "  > Test simulation ..." << std::endl;
    cross_section->SetParameters(pol_up, AN_sim, TMath::PiOver2());

    auto X0_test_up_tree = std::make_unique<TreeHelper>("X0_test_up_tree");

    X0_test_up_tree->samples(num_spin_up, weight_up, 0.0, cross_section.get());
    X0_test_up_tree->write();

    // --- Spin down ---
    cross_section->SetParameters(pol_down, AN_sim, -TMath::PiOver2());

    auto X0_test_down_tree = std::make_unique<TreeHelper>("X0_test_down_tree");

    X0_test_down_tree->samples(num_spin_down, weight_down, 0.0, cross_section.get());
    X0_test_down_tree->write();

    return 0;
}
