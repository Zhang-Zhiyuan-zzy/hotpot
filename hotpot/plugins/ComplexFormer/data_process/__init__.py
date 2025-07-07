# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : __init__.py
 Created   : 2025/7/7 13:17
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 This module for processing raw data (from ccdc, tmqm, SclogK, ...)
 to PyG.Data format with same the key signature:
    Data(
        metal_index=Tensor[1],
        metal_attr=Tensor[1, Em],
        metal_attr_names=List[Em],
        cbond_index=Tensor[2, C],
        is_cbond=Tensor[C],
        {prefix}x=Tensor[N, Ex],
        {prefix}x_names=List[Ex],
        {prefix}edge_index=Tensor[2, L],
        {prefix}edge_attr=Tensor[L, Ee],
        {prefix}edge_attr_names=List[Ee],
        {prefix}pair_index=Tensor[2, C(N,2)],
        {prefix}pair_attr=Tensor[L, Ep],
        {prefix}pair_attr_names=List[Ep],
        {prefix}mol_rings_nums=Tensor[B],
        {prefix}rings_node_index=Tensor[R],
        {prefix}rings_node_nums=Tensor[Rn],
        {prefix}mol_rings_node_nums=Tensor[B],
        {prefix}rings_attr=Tensor[R, Er],
        {prefix}rings_attr_names=List[Er],
        {prefix}y=Tensor[Y]
        {prefix}y_names=List[Y],
        {prefix}props=Tensor[P],
        {prefix}props_names=List[P]
    )

 Where:
    the {prefix} indicates the components (ignored for the complexes main),
        the {prefix} could be `sol`, `med`, ...
    B is the batch_size
    E is the embedding_size
    N is the number of nodes
    L is the number of edges
    R is the number of rings
    Y is the dims of target vector in the graph(or, mol)-level, used usually for the global target
    P is the dims of property vectors in the graph-level, used for the components like solvents(prefix='sol'), methods or other.
===========================================================
"""
