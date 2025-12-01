// Standard C++ Includes
#include <iostream>

// ROOT Includes
#include "TFile.h"
#include "TRandom3.h"
#include "TString.h"

// Your Custom Header
#include "omnifold/Simulation.hh" 

int run_simulation() {
    // 1. Centralized Configuration
    // Define physics parameters once to ensure consistency across Train/Test/Data
    const int n_samples = 200000;
    
    // Beam Parameters
    const double P_up = 0.9;
    const double P_down = 0.7;
    const double L_up = 0.6;
    const double L_down = 0.4;
    
    // Physics Parameters
    const double AN_sim = 0.0; // Simulation (No Asymmetry)
    const double AN_dat = 0.2; // Data (Asymmetry)

    // 2. Setup Output
    TFile* outfile = TFile::Open("./outputs/simulation.root", "RECREATE");
    
    // 3. Setup Generator
    TRandom3* generator = new TRandom3(42); 

    // ==========================================
    // Training Sample (X0, AN=0)
    // ==========================================
    std::cout << "[===> Creating train events (X0) ]" << std::endl;
    Simulation* X0 = new Simulation("X0");
    
    X0->set_parameters(P_up, P_down, L_up, L_down, AN_sim);
    X0->samples(n_samples, generator);
    X0->write();
    
    delete X0; // Clean up immediately after use to free RAM

    // ==========================================
    // Data/Signal Sample (X1, AN=0.2)
    // ==========================================
    std::cout << "[===> Creating data events (X1) ]" << std::endl;
    Simulation* X1 = new Simulation("X1");
    
    X1->set_parameters(P_up, P_down, L_up, L_down, AN_dat);
    X1->samples(n_samples, generator);
    X1->write();
    
    delete X1;

    // ==========================================
    // Test Sample (X0_test, AN=0)
    // ==========================================
    std::cout << "[===> Creating test events (X0_test) ]" << std::endl;
    Simulation* X0_test = new Simulation("X0_test");
    
    X0_test->set_parameters(P_up, P_down, L_up, L_down, AN_sim);
    X0_test->samples(n_samples, generator);
    X0_test->write();
    
    delete X0_test;
    
    // 4. Final Cleanup
    std::cout << "[===> Closing file: ./outputs/simulation.root ]" << std::endl;
    outfile->Close(); 
    
    delete outfile;
    delete generator;

    return 0;
}