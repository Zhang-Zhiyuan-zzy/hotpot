#include <string>
#include <filesystem>
#include <stdexcept>
#include <iostream>
#include <typeinfo>
#include <list>
#include <vector>
#include <openbabel3/openbabel/obconversion.h>
#include <openbabel3/openbabel/mol.h>
#include <openbabel3/openbabel/atom.h>

#include "../../include/OBcheminfo.h"

using namespace std;
// namespace fs = std::filesystem;


string open_file(const string& fp, const char* mode="r")
{
    // Define variables
    ifstream fileStream(fp);
    stringstream buffer;
    string fileContent;

    if (!fileStream.is_open()) {
        std::cerr << "Could not open the file: " << fp << std::endl;  
        return ""; 
    }

    buffer << fileStream.rdbuf();
    
    return buffer.str();
}

string mol2smiles(OpenBabel::OBMol mol)
{
    // Initialize Open Babel  
    OpenBabel::OBConversion conv;  
    conv.SetOutFormat("smi"); // Set the output format to SMILES

    istringstream tokenStream(conv.WriteString(&mol));
    // tokenStream.str(conv.WriteString(&mol));

    string smiles; 
    while (tokenStream >> smiles) {return smiles;}
    return smiles;
}

list<OpenBabel::OBMol> readOBMol(const string& src, string fmt = "")
{

    string content = "";
    filesystem::path src_path;

    bool file_is_exist = (filesystem::exists(src) && filesystem::is_regular_file(src));
    cout << "File is or not exist: " << file_is_exist << endl;
    if (file_is_exist) {
        cout << "File is exist!" << endl;
        cout << "the src_path is: " << src_path << endl;

        src_path = src;
        content = open_file(src_path);

    } else {
        cout << "File is not exist!" << endl;
        content = src;
    }


    // Specify the content format.
    if (fmt.empty()){
        if (file_is_exist) {
            fmt = src_path.extension();
            fmt = fmt.substr(1, fmt.length()-1);
        } 

        else{
            ostringstream msg;
            msg << "The path of '" << src << "is not exist!!!";
            throw runtime_error(msg.str());
        }
    }


    if (fmt == "log" || fmt == "g16log") {
        fmt = "g16";
    }

    cout << "The final `fmt` is: " << fmt << "." << endl;


    list<OpenBabel::OBMol> molList = {};

    if (!fmt.empty()) {

        OpenBabel::OBConversion conv;
        conv.SetInFormat(fmt.c_str());

        OpenBabel::OBMol mol;

        bool notatend = conv.ReadString(&mol, content);
        while (notatend)
        {
            cout << "The atom number of mol is: " << mol.NumAtoms() << endl;
            molList.push_back(mol);
            mol.Clear();
            notatend = conv.Read(&mol);

            cout << "The length of molList is: " << molList.size() << endl;
        }

    }

    return molList;
}

list<OpenBabel::OBMol> readOBMolFromString(const string strContent, string fmt)
{
    OpenBabel::OBConversion conv;
    conv.SetInFormat(fmt.c_str());

    list<OpenBabel::OBMol> listMol;
    OpenBabel::OBMol mol;
    bool notatend = conv.ReadString(&mol, strContent);
        while (notatend) {
        listMol.push_back(mol);
        mol.Clear();
        notatend = conv.Read(&mol);
    }

    return listMol;
}


list<OpenBabel::OBMol> readOBMolFromFile(const string& filePath, string fmt)
{
    OpenBabel::OBConversion conv;
    conv.SetInFormat(fmt.c_str());

    list<OpenBabel::OBMol> molList;
    OpenBabel::OBMol mol;
    bool notatend = conv.ReadFile(&mol, filePath);
    while (notatend) {
        molList.push_back(mol);
        mol.Clear();
        notatend = conv.Read(&mol);
    }

    return molList;
}

list<OpenBabel::OBMol> _readOBMOL(const string src, string fmt = "")
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
            msg << "The path of '" << src << "is not exist!!!";
            throw runtime_error(msg.str());
        }

        if (fmt == "log" || fmt == "g16log") {
            fmt = "g16";
        }
    }

    if (file_is_exist) {return readOBMolFromFile(src, fmt);} 
    else {return readOBMolFromString(src, fmt);}
}

// void createFromOBMol(Molecule* mol_ptr, OpenBabel::OBMol obMol)
// {
//     for (auto obAtom : obMol.BeginAtoms()) {
//         cout << "the atomic number is: " << obAtom.AtomicNum() << endl;
//     }
    
// }

// int main(int argc, char* argv[])
// {
//     string srcContent, fmt;
//     if (argc < 2) 
//     {  
//         // go into the test mode
//         // srcContent = "c1ccccc1";
//         srcContent = "/mnt/d/zhang/OneDrive/hotpot/test/input/Am_BuPh-BPPhen.log";
//         fmt = "";
//     } 
//     else {
//         srcContent = argv[1];
//         fmt = argv[2];
//     }

//     cout << "The input format is: " << fmt << endl;

//     read_mol(srcContent, fmt);

//     return 0;
// }

struct OBAtomInfo {
    int atomic_number;
    int formal_charge;
    float partial_charge;
    int spin_multiplicity;
    double mass;
    int valence;
    int hybridization;
    bool is_aromatic;
    vector<float> coords;

    // bools
    int is_chiral;
    int is_Axial;

};


// Implementation of OBMolecule
// Constructors of OBMolecule
OBMolecule::OBMolecule() : OBMol() {}  // Default constructor
OBMolecule::OBMolecule(const OBMolecule &other) : OBMol(other) {}
OBMolecule::OBMolecule(const OpenBabel::OBMol &obMol) : OBMol(obMol) {}

// Destructor
// OBMolecule::~OBMolecule() {}


int main()
{   
    string src = "/mnt/d/zhang/OneDrive/hotpot/cpp/data/Compound_127500001_128000000.sdf";
    OpenBabel::OBConversion conv;
    conv.SetInFormat("sdf");

    OpenBabel::OBMol obMol;
    bool notatend = conv.ReadFile(&obMol, src);
    while (notatend)
    {
        // cout << "The atom number of " << obMol.GetFormula() << "is: " << obMol.NumAtoms() << endl;
        obMol.Clear();
        notatend = conv.Read(&obMol);
    }

    cout << "Read is end!!!!!" << endl;

    conv.SetOutputIndex(0);
    obMol.Clear();
    notatend = conv.Read(&obMol);

    while (notatend)
    {
        cout << "The atom number of " << obMol.GetFormula() << "is: " << obMol.NumAtoms() << endl;
        obMol.Clear();
        notatend = conv.Read(&obMol);
    }

    cout << "Read is end!!!!!" << endl;

    return 0;
}
