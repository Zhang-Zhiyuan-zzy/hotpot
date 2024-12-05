#include <iostream>
#include <typeinfo>
#include <openbabel3/openbabel/obconversion.h>
#include <openbabel3/openbabel/mol.h>

using namespace std;


int main(int argc,char **argv)
{
    cout << "The argc is: " << argc << endl;
    cout << "The point of argv is:" << argv[1] << endl;
    cout << "Library path: " << __FILE__ << endl;

    OpenBabel::OBConversion obconversion;
    obconversion.SetInFormat("smi");
    OpenBabel::OBMol mol;

    // std::cout << "The type of obcomversion is: " << typeid(obconversion) << std::endl;
    // std::cout << "The type of mol is: " << typeid(mol) << std::endl;
    int a = 1;
    const char* c = "a";
    string s = "Hello World";
    cout << "The type name of obcomversion is: " << typeid(obconversion).name() << endl;
    cout << "The type name of mol is: " << typeid(mol).name() << endl;
    cout << "The type name of 1 is: " << typeid(1).name() << endl;
    cout << "The type name of 2 is: " << typeid(2).name() << endl;
    cout << "The type name of a is: " << typeid(a).name() << endl;
    cout << "The type name of c is: " << typeid(c).name() << endl;
    cout << "The type name of s is: " << typeid(s).name() << endl;

    if (argc < 2) {
        cerr << "The smiles of molecule should be given!!!!";
        return 1;
    }

    bool notatend = obconversion.ReadString(&mol, argv[1]);
    while (notatend)
    {
        cout << "Molecular Weight: " << mol.GetMolWt() << endl;

        mol.Clear();
        notatend = obconversion.Read(&mol);
    }

    return(0);
}
