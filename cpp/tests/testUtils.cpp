#include <string>
#include <tuple>
#include <vector>
#include <typeinfo>
#include <filesystem>
#include <exception>
#include <memory>
#include "../include/utils.h"


using namespace std;
using namespace OpenBabel;


// Define work dirs
filesystem::path getSourceDirectory() {return filesystem::path(__FILE__).parent_path();} // Get the directory of the current file 
filesystem::path getHpCppRoot() {return getSourceDirectory().parent_path();}
filesystem::path pathMolDatabase() {return getHpCppRoot() / "data" / "Compound_127500001_128000000.sdf";}


void testReader() {
    const filesystem::path testFilePath = pathMolDatabase();
    const std::string format = ""; // Set to an actual expected format

    cout << "The testFilePath is: " << testFilePath << endl;

    try {  
        IoFunc::OBMolReader reader(testFilePath.string(), format);  
        optional<OBMol> result;

        while ((result = reader.Read()), result) {cout << "result value " << result.has_value() << endl;}
        cout << "End of Reader!!!!!!!!!!!" << endl;

        reader.Refresh();
        for (auto result : reader) {cout << "The number of atoms of "<< result.GetFormula() << " is: " << result.NumAtoms() << endl;}

        // test get atom individually by pointers
        OBAtomIterator atomIter = result -> BeginAtoms();
        OBAtom* atom = *atomIter;
        cout << "Atomic number is: " << atom -> GetAtomicNum() << endl;
        atomIter++;
        atom = *atomIter;
        cout << "Atomic number is: " << atom -> GetAtomicNum() << endl;

        for (OBAtomIterator atom_ptr = result -> BeginAtoms(); atom_ptr != result -> EndAtoms(); atom_ptr++) {
            cout << "Atomic number is: " << (*atom_ptr) -> GetAtomicNum() << endl;
        }

        reader.Refresh();
        result = reader.Read();
        
        cout << "The type of result is " << typeid(result).name() << endl;
        cout << "The result is " << result.has_value() << "values" << endl;
        if (!result.has_value()) {  
            std::cerr << "Test failed: MolReader did not return a valid OBMol." << std::endl;  
            exit(1);  
        }  
        if (result->Empty()) {  
            std::cerr << "Test failed: Read OBMol is empty." << std::endl;  
            exit(1);  
        }  

        std::cout << "MolReader read test passed." << std::endl;  

    } catch (const std::runtime_error& e) {  
        std::cerr << "Test failed in testReader(): " << e.what() << std::endl;  
        exit(1);  
    }  
}

void testBuild3D() {
    const filesystem::path testFilePath = pathMolDatabase();
    const std::string format = ""; // Set to an actual expected format

    try {
        IoFunc::OBMolReader reader(testFilePath.string(), format);
        auto mol = *reader.Read();
        vector<vector3> coords;

        cout << "The dimension of OBMol is: " << mol.Has3D() << endl;
        Conformer::build3D(&mol);
        cout << "The dimension of OBMol is: " << mol.Has3D() << endl;

        coords = Conformer::Get3DVectors(mol);
        cout << "The size of vector is: " << coords.size() << endl;

        ChemTools::printOBMolCoordinates(mol);


        // int count = 0;
        // int total_coords = (mol->NumAtoms()) * 3 + 10;
        // double* coord_ptr = mol->GetCoordinates();
        // while (count < total_coords) {
        //     cout << "The first row and first atom value of coordinates is: " << *coord_ptr << endl;
        //     coord_ptr++;
        //     count++;
        // }

        // cout << "Number of coordinates are: " << count << endl;

    }
    catch (const std::runtime_error& e) {
        std::cerr << "Test failed in testBuild3D(): " << e.what() << std::endl;  
        exit(1);  
    }
}


int main() {
    // testReader();
    testBuild3D();
    return 0;
}
