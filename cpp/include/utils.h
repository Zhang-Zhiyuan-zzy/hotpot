#ifndef HPUTILS_H  // Include guard  
#define HPUTILS_H

#include <vector>
#include <string>
#include <tuple>
#include <openbabel3/openbabel/obconversion.h>
#include <openbabel3/openbabel/mol.h>
#include <openbabel3/openbabel/builder.h>
#include <openbabel3/openbabel/forcefield.h>


using namespace std;
using namespace OpenBabel;

enum class ForceFieldName {
    GAFF,
    Ghemical,
    MMFF94,
    MMFF94s,
    UFF,
};

namespace ChemTools {
    void printOBMolCoordinates(OBMol mol);
}

namespace Conformer {
    vector<vector3> Get3DVectors(OBMol mol);
    void build3D(OBMol* mol, const string ff_name = "UFF", const int steps=500);
}


namespace IoFunc {

    // bool _isFile(string src);
    // OBMol readOneMol(string src, string fmt="", int which=0);
    // vector<OBMol> readRangeMols(string src, string fmt="", long start=0, long end=-1);
    // vector<OBMol> readAllMols(string src, string fmt="");

    class OBMolReader
    {
    protected:
        string _format;
        string _srcContent;
        bool _fileExist;
        OBConversion _conv;
        OBMol _obMol;
        bool _notatend;

        void _initReader();

        // Declaration of OBMolReader Iterator;
        class Iterator
        {
        private:
            OBMolReader& reader;
            optional<OBMol> mol_ptr;
        
        public:
            // Iterator constructor
            Iterator(OBMolReader& reader);

            // Operators
            OBMol& operator*();
            Iterator operator++();
            bool operator!=(Iterator& other);
        };  // end OBMolReader::Iterator


    public:
        // Constructors
        OBMolReader(string src, string fmt="");

        // Readers
        optional<OBMol> Read();
        void Refresh();

        // Iterator methods
        Iterator begin();
        Iterator end();

    };  // end class OBMolReader
}


#endif  // HPUTILS_H
