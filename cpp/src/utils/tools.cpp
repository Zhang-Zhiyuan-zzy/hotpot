#include "../../include/utils.h"

#include <openbabel3/openbabel/obconversion.h>
#include <openbabel3/openbabel/mol.h>
#include <openbabel3/openbabel/builder.h>
#include <openbabel3/openbabel/forcefield.h>
#include <openbabel3/openbabel/math/vector3.h>
#include <openbabel3/openbabel/obiter.h>

using namespace OpenBabel;

void ChemTools::printOBMolCoordinates(OBMol mol) {
    string symbol;
    double x, y, z;
    OBAtom* atom_ptr;
    cout << "Molecule " << mol.GetFormula() << " Coordinates is: " << endl;
    for (OBAtomIterator atomIter = mol.BeginAtoms(); atomIter!=mol.EndAtoms(); atomIter++) {
        atom_ptr = *atomIter;
        symbol = atom_ptr->GetType();
        x = atom_ptr->x();
        y = atom_ptr->y();
        z = atom_ptr->z();
        cout << "Atom("<< symbol <<"):" << " [" << x << ", " << y << ", " << z << "]" << endl;
    }
}