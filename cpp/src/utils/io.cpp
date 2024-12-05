#include <filesystem>
#include <stdexcept>
#include <typeinfo>
#include <memory>
#include <openbabel3/openbabel/mol.h>
#include <openbabel3/openbabel/obconversion.h>
#include <openbabel3/openbabel/forcefield.h>
#include <openbabel3/openbabel/builder.h>

#include "../../include/utils.h"

using namespace std;
using namespace OpenBabel;


// IoFunc::_isFile(string src) {

// }


// IoFunc::readOneMol(string src, string fmt, int which) {
//     OBConversion _conv;
//     _conv.SetInFormat(fmt.c_str());

//     OBMol mol;
//     notatend = _conv.ReadFile(&OBMol, src);

//     if (notatend) {return mol;}
//     else {
//         cerr << ""
//     }

// }

void IoFunc::OBMolReader::_initReader()
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


IoFunc::OBMolReader::OBMolReader(string src, string fmt)
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
void IoFunc::OBMolReader::Refresh() {this -> _initReader();}

optional<OBMol> IoFunc::OBMolReader::Read()
{
    if (!(this -> _notatend) || this -> _obMol.Empty()) {return nullopt;}

    OBMol mol(this -> _obMol);

    this -> _obMol.Clear();
    this -> _notatend = this -> _conv.Read(&(this -> _obMol));

    return mol;
}

// Constructor
IoFunc::OBMolReader::Iterator::Iterator(IoFunc::OBMolReader& reader) : reader(reader)
{
    this -> reader = reader;
    (this -> reader).Refresh();
    this -> mol_ptr = (this -> reader).Read();
}

OBMol& IoFunc::OBMolReader::Iterator::operator*() {return *(this -> mol_ptr);}
IoFunc::OBMolReader::Iterator IoFunc::OBMolReader::Iterator::operator++() 
{
    this -> mol_ptr = (this -> reader).Read();
    return *this;  // Return IoFunc::OBMolReader::Iterator instance.
}

IoFunc::OBMolReader::Iterator IoFunc::OBMolReader::begin() {return IoFunc::OBMolReader::Iterator(*this);}
IoFunc::OBMolReader::Iterator IoFunc::OBMolReader::end() {return IoFunc::OBMolReader::Iterator(*this);}

// If got a new optional<OBMolecule> return true ELSE false.
bool IoFunc::OBMolReader::Iterator::operator!=(IoFunc::OBMolReader::Iterator& other) {return this -> mol_ptr.has_value();}
