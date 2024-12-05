#include <vector>
#include <tuple>
#include <openbabel3/openbabel/obconversion.h>
#include <openbabel3/openbabel/mol.h>
#include <openbabel3/openbabel/builder.h>
#include <openbabel3/openbabel/forcefield.h>
#include <openbabel3/openbabel/math/vector3.h>
#include <openbabel3/openbabel/obiter.h>

#include "../../include/utils.h"

using namespace std;
using namespace OpenBabel;


vector<vector3> Conformer::Get3DVectors(OBMol mol) {
    vector<vector3> coords;
    // vector3<double> a_vec;
    OBAtom a;
    FOR_ATOMS_OF_MOL (a, mol) {
        // a_vec = {a -> GetX(), a -> GetY(), a -> GetZ()};
        coords.push_back(a -> GetVector());
    }

    return coords;
}

// Build 3D structure for a given molecule based on ForceField.
void Conformer::build3D(OBMol* mol_ptr, string ff_name, int steps) {
    OBBuilder _builder;
    _builder.Build(*mol_ptr);

    // Select the forcefield, this returns a pointer that we
    // will later use to access the forcefield functions.
    OBForceField* pFF = OBForceField::FindForceField(ff_name);

    // Set the logfile (can also be &cout or &cerr)
    pFF->SetLogFile(&cerr);
    pFF->SetLogLevel(0);

    // We need to setup the forcefield before we can use it. Setup()
    // returns false if it failes to find the atom types, parameters, ...
    if (!pFF->Setup(*mol_ptr)) {cerr << "ERROR: could not setup force field." << endl;}

    pFF -> ConjugateGradients(steps);
}