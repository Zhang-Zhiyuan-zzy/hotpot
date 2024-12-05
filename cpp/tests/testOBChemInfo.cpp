#include <iostream>  
#include <stdexcept>
#include <filesystem>  

#include "../include/OBcheminfo.h"  
#include <openbabel3/openbabel/mol.h>
#include <openbabel3/openbabel/forcefield.h>

using namespace std;

// Define work dirs
filesystem::path getSourceDirectory() {return filesystem::path(__FILE__).parent_path();} // Get the directory of the current file 
filesystem::path getHpCppRoot() {return getSourceDirectory().parent_path();}
filesystem::path pathMolDatabase() {return getHpCppRoot() / "data" / "Compound_127500001_128000000.sdf";}

void test_OBMoleculeConstruction() {  
    OBMolecule mol1;  // Default constructor  

    // Test that the OBMolecule is not empty  
    if (mol1.Empty()) {  
        std::cerr << "Test failed: Default constructed OBMolecule is empty." << std::endl;  
        exit(1); // Indicate failure  
    }  

    // Copy constructor test  
    OBMolecule mol2(mol1);  
    if (mol2.Empty()) {  
        std::cerr << "Test failed: Copied OBMolecule is empty." << std::endl;  
        exit(1);  
    }  
    
    // Create an OBMol object and test the constructor with it  
    OpenBabel::OBMol baseMol;  
    OBMolecule mol3(baseMol);  
    if (mol3.Empty()) {  
        std::cerr << "Test failed: OBMolecule constructed from OBMol is empty." << std::endl;  
        exit(1);  
    }  

    std::cout << "OBMolecule construction tests passed." << std::endl;  
}

void test_MolReader() {  
    const filesystem::path testFilePath = pathMolDatabase();
    const std::string format = ""; // Set to an actual expected format

    cout << "The testFilePath is: " << testFilePath << endl;

    try {  
        Io::MolReader reader(testFilePath.string(), format);  
        optional<OBMolecule> result;

        while ((result = reader.Read()), result) {}
        cout << "End of Reader!!!!!!!!!!!" << endl;
        reader.Refresh();
        for (auto result : reader) {cout << "The number of atoms of "<< result.GetFormula() << " is: " << result.NumAtoms() << endl;}

        int count = 0;
        OBAtomIterator atomIter = result -> BeginAtoms();
        OBAtom* atom = *atomIter;
        // for (OBAtomIterator atom = result -> BeginAtoms(); atom != result -> EndAtoms(); atom++) {
        //     cout << "The" << count << "-th atom is: " << atom -> GetAtomicNum() << endl;
        //     count++;
        // }
        cout << "Atomic number is: " << atom -> GetAtomicNum() << endl;
        atomIter++;
        atom = *atomIter;
        cout << "Atomic number is: " << atom -> GetAtomicNum() << endl;

        for (OBAtomIterator atom_ptr = result -> BeginAtoms(); atom_ptr != result -> EndAtoms(); atom_ptr++) {
            cout << "Atomic number is: " << (*atom_ptr) -> GetAtomicNum() << endl;
        }
        
        if (!result.has_value()) {  
            std::cerr << "Test failed: MolReader did not return a valid OBMolecule." << std::endl;  
            exit(1);  
        }  
        if (result->Empty()) {  
            std::cerr << "Test failed: Read OBMolecule is empty." << std::endl;  
            exit(1);  
        }  

        std::cout << "MolReader read test passed." << std::endl;  

    } catch (const std::runtime_error& e) {  
        std::cerr << "Test failed: " << e.what() << std::endl;  
        exit(1);  
    }  
}  


int main() {  
    // Run tests  
    test_MolReader();  

    std::cout << "All tests passed!" << std::endl;  
    return 0;  
}
