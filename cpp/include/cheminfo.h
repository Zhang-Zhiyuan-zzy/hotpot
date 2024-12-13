#ifndef CHEMINFO_H  // Include guard  
#define CHEMINFO_H

#include <string>
#include <vector>
#include <memory>
#include <openbabel3/openbabel/mol.h>

using namespace std;

// Base variables
namespace BaseVar {
    const string atomicSymbol[121];

}

// Base functions
namespace BaseFunc {
    string getAtomicSymbol(const int atomicNum);
    int    getAtomicNumber(const string atomSymbol);
}


class Atom {
private:
    int _atomic_number;
    int _formal_charges;
    float _partial_charges;
    string _symbol;
    vector<float> _coords;

public:
    // Constructors
    Atom(
        const int atomic_number, 
        const int formal_charges = 0, 
        const float partial_charge = 0.,
        const vector<float> = {0., 0., 0.}
    );

    Atom(
        const string symbol, 
        const int formal_charges = 0, 
        const float partial_charge = 0.,
        const vector<float> = {0., 0., 0.}
    );

    // Destructor
    ~Atom();

    // Getters
    int getAtomicNumber();
    string getSymbol();
    vector<int> getCoordinates();
    int getFormalCharges();
    float getPartialCharges();

};

class Bond {
private:
    shared_ptr<Atom> _atom1_ptr, _atom2_ptr;
    int _bond_order;
    enum BondType {Unknown, Single, Double, Triple, Aromatic, Delocalized, Metalic};

public:
    // Constructors
    Bond(Atom atom1, Atom atom2);
    Bond(shared_ptr<Atom> atom1_ptr, shared_ptr<Atom> atom2_ptr);

    // Destructor
    ~Bond();
};

class Molecule {  
private:  
    vector<shared_ptr<Atom>> _atoms_ptr;
    vector<shared_ptr<Atom>> _bonds_ptr;

public:

    int charges;

    // Constructors  
    Molecule();  // Default constructor  
    Molecule(OpenBabel::OBMol obMol);  // construct from OpenBabel OBMol

    // Destructor  
    ~Molecule();  

    // Readers
    void readFile(const string filePath, string fmt = "");
    void readString(const string contentString, string fmt = "");
    void readFromOBMol(OpenBabel::OBMol obMol);

    // Setters  
    void addAtom(Atom atom);
    void addAtom(shared_ptr<Atom> atom_ptr);

    void addBond(Bond bond);
    void addBond(shared_ptr<Bond> bond_ptr);

    // Getters  
    vector<Atom> getAtoms();
    vector<Bond> getBonds();
    string getSmiles();

    // Method to display molecule information  
    void displayInfo() const;  
};  

#endif  // CHEMINFO_H
