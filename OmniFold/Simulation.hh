#include <iostream>
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TRandom3.h"
#include "TF1.h"
#include "TMath.h"
#include "TString.h"

class Simulation {
private:
    TF1* cross_section;
    const double pi = TMath::Pi(); 

public:
    // Public Members
    double P[2];
    double L[2];
    double phi_pol[2];
    double AN;
    
    // Branch Variables
    double X;
    double W;
    double Y;

    // ROOT Objects
    TTree* TreeSpinUp = nullptr;
    TTree* TreeSpinDown = nullptr;
    TH1D* HistSpinUp = nullptr;
    TH1D* HistSpinDown = nullptr;

    // Methods
    Simulation(TString fname);
    virtual ~Simulation();
    void set_parameters(double P_up, double P_down, double L_up, double L_down, double AN_val);
    void samples(int n_samples, TRandom3* generator);
    void write();
};

// ==========================================
// Implementation
// ==========================================

Simulation::Simulation(TString fname) {
    TreeSpinUp = new TTree(Form("TreeSpinUp_%s", fname.Data()), "Spin Up Tree");
    TreeSpinUp->Branch("X", &X, "X/D");
    TreeSpinUp->Branch("W", &W, "W/D");

    TreeSpinDown = new TTree(Form("TreeSpinDown_%s", fname.Data()), "Spin Down Tree");
    TreeSpinDown->Branch("X", &X, "X/D");
    TreeSpinDown->Branch("W", &W, "W/D");

    // Initialize Cross Section
    // Logic: (1/2pi) * (1 + P * AN * sin(phi - x))
    cross_section = new TF1("cross_section", "(1.0/(2.0* TMath::Pi()))* (1.0 + [0]* [1]* TMath::Sin([2] - x))", -pi, pi);

    HistSpinUp = new TH1D(Form("HistSpinUp_%s", fname.Data()), ";#phi;Counts", 12, -pi, pi);
    HistSpinDown = new TH1D(Form("HistSpinDown_%s", fname.Data()), ";#phi;Counts", 12, -pi, pi);
    
    // Initialize default scalars to 0
    AN = 0;
    for(int i=0; i<2; ++i) { P[i]=0; L[i]=0; phi_pol[i]=0; }
}

Simulation::~Simulation() {
    // Cleanup to prevent memory leaks
    if (cross_section) delete cross_section;
    if (TreeSpinUp) delete TreeSpinUp;
    if (TreeSpinDown) delete TreeSpinDown;
    if (HistSpinUp) delete HistSpinUp;
    if (HistSpinDown) delete HistSpinDown;
}

void Simulation::set_parameters(double P_up, double P_down, double L_up, double L_down, double AN_val) {
    P[0] = P_up;
    P[1] = P_down;

    L[0] = L_up;
    L[1] = L_down;
    
    this->AN = AN_val;
}

void Simulation::samples(int n_samples, TRandom3* generator) {
    
    // Configuration
    phi_pol[0] = pi/2.0;  // Usually Spin Up is +Pi/2 or 0
    phi_pol[1] = -pi/2.0; // Usually Spin Down is -Pi/2 or Pi

    // --- Spin Up Generation ---
    int n_spin_up = (int)(L[0] * n_samples);
    std::cout << "[ ===> Creating " << n_spin_up << " events with Spin Up config ]" << std::endl;

    cross_section->SetParameters(P[0], AN, phi_pol[0]);

    std::cout << "P : " << cross_section->GetParameter(0) << std::endl;
    std::cout << "AN : " << cross_section->GetParameter(1) << std::endl;
    std::cout << "phi_pol : " << cross_section->GetParameter(2) << std::endl;

    for(int i = 0; i < n_spin_up; i++) {
        X = cross_section->GetRandom(-pi, pi, generator);
        
        // Weight calculation
        double denom = 2.0 * L[0] * P[0];
        if (denom == 0) W = 1.0; // Safety check
        else W = (L[0]* P[0] + L[1]* P[1]) / denom;

        HistSpinUp->Fill(X, W);
        TreeSpinUp->Fill();
    }

    // --- Spin Down Generation ---
    int n_spin_down = (int)(L[1] * n_samples);
    std::cout << "[ ===> Creating " << n_spin_down << " events with Spin Down config ]" << std::endl;

    cross_section->SetParameters(P[1], AN, phi_pol[1]);

    std::cout << "P : " << cross_section->GetParameter(0) << std::endl;
    std::cout << "AN : " << cross_section->GetParameter(1) << std::endl;
    std::cout << "phi_pol : " << cross_section->GetParameter(2) << std::endl;

    for(int i = 0; i < n_spin_down; i++) {
        X = cross_section->GetRandom(-pi, pi, generator);
        
        double denom = 2.0 * L[1] * P[1];
        if (denom == 0) W = 1.0;
        else W = (L[0]* P[0] + L[1]* P[1]) / denom;

        HistSpinDown->Fill(X, W);
        TreeSpinDown->Fill();
    }

    // HistSpinUp->Scale(1.0/HistSpinUp->Integral());
    // HistSpinDown->Scale(1.0/HistSpinDown->Integral());
}

void Simulation::write() {
    if(TreeSpinUp) TreeSpinUp->Write();
    if(TreeSpinDown) TreeSpinDown->Write();
    if(HistSpinUp) HistSpinUp->Write();
    if(HistSpinDown) HistSpinDown->Write();
}