#include "../../include/cheminfo.h"

// Implementation of Atom
Atom::Atom(
    const int atomicNum, 
    const int formalCharges = 0, 
    const float partialCharges = 0.,
    const vector<float> Coords = {0., 0., 0.}
) {
    _atomic_number = atomicNum;
    _symbol = BaseFunc::getAtomicSymbol(atomicNum);
    _formal_charges = formalCharges;
    _partial_charges = partialCharges;
    _coords = Coords;
}
