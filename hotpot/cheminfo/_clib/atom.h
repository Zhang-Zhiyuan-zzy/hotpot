//
// Created by zhang on 2025/1/27.
//

#ifndef ATOM_H
#define ATOM_H

#include <stdbool.h>

extern bool is_metal(int atomic_number);
extern int calc_implicit_hydrogens(int atomic_number, bool is_aromatic, int *neigh_atoms, int neigh_length);
extern int get_valence(int atomic_number, int *neigh_atoms, int neigh_length);
extern int sum_list(int *neigh_atoms, int neigh_length);

#endif //ATOM_H
