#include <string>
#include <stdexcept>
#include "../../include/cheminfo.h"

using namespace std;


string BaseVar::atomicSymbol[121] = {
        "0",
        "H" ,"He","Li","Be","B" ,"C" ,"N" ,"O" ,"F" ,"Ne",
        "Na","Mg","Al","Si","P" ,"S" ,"Cl","Ar","K" ,"Ca",
        "Sc","Ti","V" ,"Cr","Mn","Fe","Co","Ni","Cu","Zn",
        "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y" ,"Zr",
        "Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn",
        "Sb","Te","I" ,"Xe","Cs","Ba","La","Ce","Pr","Nd",
        "Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb",
        "Lu","Hf","Ta","W" ,"Re","Os","Ir","Pt","Au","Hg",
        "Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th",
        "Pa","U" ,"Np","Pu","Am","Cm","Bk","Cf","Es","Fm",
        "Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds",
        "Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og",""  ,""
};
size_t _symbolCounts = sizeof(BaseVar::atomicSymbol) / sizeof(BaseVar::atomicSymbol);


string BaseFunc::getAtomicSymbol(const int atomicNum) {
    if (atomicNum >= 1 && atomicNum <= 120) {  
        return BaseVar::atomicSymbol[atomicNum];  
    } else {  
        throw out_of_range("Atomic number must be between 1 and 120."); 
    }
}


int BaseFunc::getAtomicNumber(const string atomicSymbol) {
    
    for (int i = 0; i < _symbolCounts; ++i) {  
        if (BaseVar::atomicSymbol[i] == atomicSymbol) {  
            return i; // Return the index of the first occurrence  
        }  
    }  
    return -1; // Return -1 if the value is not found  
}


// string getSymbol(const int atomicNumber) {
//     if (atomicNumber >= 1 && atomicNumber <= 120) {  
//         return atomicSymbol[atomicNumber];  
//     } else {  
//         throw out_of_range("Atomic number must be between 1 and 120.");  
//     } 
// }
