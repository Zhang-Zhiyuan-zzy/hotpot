#ifndef OBCHEMINFO_H  // Include guard  
#define OBCHEMINFO_H

#include <string>
#include <variant>
#include <optional>  
#include <openbabel3/openbabel/mol.h>
#include <openbabel3/openbabel/obconversion.h>

using namespace std;
using namespace OpenBabel;

class OBMolecule : public OBMol
{
public:

    // Constructors
    OBMolecule();
    OBMolecule(const OBMolecule &other);  // construct by copy other one
    OBMolecule(const OBMol &obMol); // construct from Open Babel OBMol instance

    // Destructor
    virtual ~OBMolecule() {};

    // Methods
    void Build3D(string ff_name="UFF", int steps=500);  // Select from "MMFF94", "UFF", ""

    // Output
    string Smiles();
    void Write(string fmt);
    void WriteFile(string fmt, string pf);

};

namespace Io 
{

    class MolReader
    {
    protected:
        string _format;
        string _srcContent;
        bool _fileExist;
        OBConversion _conv;
        OBMol _obMol;
        bool _notatend;

        void _initReader();

        // Declaration of MolReader Iterator;
        class Iterator
        {
        private:
            MolReader& reader;
            optional<OBMolecule> mol_ptr;
        
        public:
            // Iterator constructor
            Iterator(MolReader& reader);

            // Operators
            OBMolecule& operator*();
            Iterator operator++();
            bool operator!=(Iterator& other);
        };  // end MolReader::Iterator


    public:
        // Constructors
        MolReader(string src, string fmt="");

        // Readers
        optional<OBMolecule> Read();
        void Refresh();

        // Iterator methods
        Iterator begin();
        Iterator end();

    };  // end class MolReader

}  // end namespace Io

#endif  // OBCHEMINFO_H
