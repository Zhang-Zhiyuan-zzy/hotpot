%module atom

%{
/* Your C header(s) go here. Replace "atom.h" with your actual header if available. */
#include "atom.h"
%}

/* Let SWIG know about any custom types if needed (e.g., bool).
   By default, SWIG will map C99 _Bool/bool to Python booleans. */

/* SWIG typemap to allow Python list -> C array */
%include "typemaps.i"
%apply (int *IN_ARRAY, int LENGTH) { (int *neigh_atoms, int neigh_length) };

/* Expose the is_metal() function. */
%include "atom.h"
extern bool is_metal(int atomic_number);

/* Expose the calc_implicit_hydrogens() function. */
extern int calc_implicit_hydrogens(int atomic_number, bool is_aromatic, int *neigh_atoms, int neigh_length);
extern int get_valence(int atomic_number, int *neigh_atoms, int neigh_length);
extern int sum_list(int *neigh_atoms, int neigh_length);




