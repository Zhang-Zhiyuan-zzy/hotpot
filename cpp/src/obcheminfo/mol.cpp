#include <filesystem>
#include <stdexcept>
#include <typeinfo>
#include <memory>
#include <openbabel/mol.h>
#include <openbabel/obconversion.h>
#include <openbabel/forcefield.h>
#include <openbabel/builder.h>

#include "../../include/OBcheminfo.h"

using namespace OpenBabel;


void Io::MolReader::_initReader()
{
    this -> _conv.SetInFormat((this -> _format).c_str());

    (this -> _obMol).Clear();
    if (this -> _fileExist) {
        this -> _notatend = this -> _conv.ReadFile(&(this -> _obMol), this -> _srcContent);
    } 
    else {
        this -> _notatend = this -> _conv.ReadString(&(this -> _obMol), this -> _srcContent);
    }
}


Io::MolReader::MolReader(string src, string fmt)
{
    // Determine whether the given source is a path
    bool file_is_exist = (filesystem::exists(src) && filesystem::is_regular_file(src));

    // If fmt is not given, auto specify it according to file extensive
    if (fmt.empty()){
        if (file_is_exist) {
            filesystem::path src_path(src);
            fmt = src_path.extension();
            fmt = fmt.substr(1, fmt.length()-1);
        }

        else{
            ostringstream msg;
            msg << "The path of '" << src << "' is not exist!!!";
            throw runtime_error(msg.str());
        }

        if (fmt == "log" || fmt == "g16log") {
            fmt = "g16";
        }
    }

    // Recording the input stream;
    this -> _format = fmt;
    this -> _srcContent = src;
    this -> _fileExist = file_is_exist;

    this -> _initReader();

}

// Refresh the MolReader
void Io::MolReader::Refresh() {this -> _initReader();}

Io::MolReader::Iterator Io::MolReader::begin() {return Io::MolReader::Iterator(*this);}
Io::MolReader::Iterator Io::MolReader::end() {return Io::MolReader::Iterator(*this);}


optional<OBMolecule> Io::MolReader::Read()
{
    if (!(this -> _notatend) || this -> _obMol.Empty()) {return nullopt;}

    OBMolecule mol(this -> _obMol);
    this -> _obMol.Clear();

    this -> _notatend = this -> _conv.Read(&(this -> _obMol));

    return mol;
}

// Constructor
Io::MolReader::Iterator::Iterator(Io::MolReader& reader) : reader(reader)
{
    this -> reader = reader;
    (this -> reader).Refresh();
    this -> mol_ptr = (this -> reader).Read();
}

OBMolecule& Io::MolReader::Iterator::operator*() {return *(this -> mol_ptr);}
Io::MolReader::Iterator Io::MolReader::Iterator::operator++() 
{
    this -> mol_ptr = (this -> reader).Read();
    return *this;  // Return Io::MolReader::Iterator instance.
}

// If got a new optional<OBMolecule> return true ELSE false.
bool Io::MolReader::Iterator::operator!=(Io::MolReader::Iterator& other) {return this -> mol_ptr.has_value();}

OBMolecule::OBMolecule() {}
OBMolecule::OBMolecule(const OBMolecule &other) : OBMol(other) {}
OBMolecule::OBMolecule(const OBMol &obMol) : OBMol(obMol) {}

void OBMolecule::Build3D(string ff_name, int steps)
{

    OBBuilder _builder;
    _builder.Build(*this);

    shared_ptr<OBForceField> ff_ptr(OBForceField::FindForceField(ff_name));
    if (!ff_ptr) {cerr << "ERROR: got an empty force field." << endl;}

    ff_ptr -> SetLogFile(&cerr);
    ff_ptr -> SetLogLevel(OBFF_LOGLVL_LOW);

    if (!ff_ptr -> Setup(*this)) { cerr << "ERROR: could not setup force field." << endl;}

    ff_ptr -> ConjugateGradients(1000);
}
