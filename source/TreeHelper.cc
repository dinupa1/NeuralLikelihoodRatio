#include "TreeHelper.hh"
#include <iostream>

TreeHelper::TreeHelper(TString name) {
    _tree = new TTree(name.Data(), "spin tree");

    _hist = new TH1D(Form("h_%s", name.Data()), "; #phi [rad]; counts", 30, -_Pi, _Pi);

    reset();
    branches();
}

TreeHelper::~TreeHelper() {
    if(_tree) delete _tree;
    if(_hist) delete _hist;
}

void TreeHelper::reset() {
    _X = _SAJ;
    _W = _SAJ;
    _Y = _SAJ;
}

void TreeHelper::branches() {
    _tree->Branch("X", &_X, "X/D");
    _tree->Branch("W", &_W, "W/D");
    _tree->Branch("Y", &_Y, "Y/D");
}

void TreeHelper::samples(int num_samples, double weight, double label, TF1* cross_section) {

    std::cout << " > Pol. " << cross_section->GetParameter(0) << std::endl;
    std::cout << "  > AN " << cross_section->GetParameter(1) << std::endl;
    std::cout << "  > phi Pol. " << cross_section->GetParameter(2) << std::endl;

    for(int i = 0; i < num_samples; i++) {
        _X = cross_section->GetRandom(-_Pi, _Pi);
        _W = weight;
        _Y = label;

        _hist->Fill(_X, _W);
        _tree->Fill();
    }
}

void TreeHelper::write() {
    if(_tree) _tree->Write();
    if(_hist) _hist->Write();
}
