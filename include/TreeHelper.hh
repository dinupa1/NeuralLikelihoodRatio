#ifndef TREEEHELPER_HH
#define TREEEHELPER_HH

#include "TTree.h"
#include "TMath.h"
#include "TF1.h"
#include "TH1D.h"
#include "TString.h"

// --- Simple helper class to save data to a tree ---
class TreeHelper {
    double _X;
    double _W;
    double _Y;
    TH1D* _hist;
    TTree* _tree;
    double _SAJ = -999.0;
    const double _Pi = TMath::Pi();
public:
    TreeHelper(TString name);
    virtual ~TreeHelper();
    void reset();
    void branches();
    void samples(int num_samples, double weight, double label, TF1* cross_section);
    void write();
};

#endif
