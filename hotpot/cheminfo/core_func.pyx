"""
python v3.9.0
@Project: hotpot
@File   : core_func
@Auther : Zhiyuan Zhang
@Data   : 2025/1/4
@Time   : 9:35
"""
import cython

from cpython cimport array
import array

cdef int* _default_valence = [0,  # 0
        1, 0, 1, 2, 3, 4, 3, 2, 1, 0,     #  1..10
        1, 2, 3, 4, 3, 2, 1, 0, 1, 2,     # 11..20
        3, 4, 5, 3, 2, 2, 3, 2, 2, 2,     # 21..30
        3, 4, 3, 2, 1, 0, 1, 2, 3, 4,     # 31..40
        5, 6, 7, 4, 3, 2, 1, 2, 3, 4,     # 41..50
        3, 2, 1, 0, 1, 2, 3, 3, 3, 3,     # 51..60
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,     # 61..70
        3, 4, 5, 6, 5, 4, 3, 2, 1, 2,     # 71..80
        3, 4, 3, 2, 1, 0, 1, 2, 3, 4,     # 81..90
        5, 6, 5, 6, 3, 3, 3, 3, 3, 3,     # 91..100
        3, 3, 3, 4, 5, 6, 7, 4, 3, 4,     # 101..110
        1, 2, 3, 4, 3, 2, 1, 0            # 111..118
]

cpdef int get_default_valence(int i):
    return _default_valence[i]


# Element categorize in periodic tabel
cdef set _alkali_metals = {3, 11, 19, 37, 55, 87}  # Group 1
cdef set _alkaline_earth_metals = {4, 12, 20, 38, 56, 88}  # Group 2
cdef set _transition_metals = set(range(21, 31)) | set(range(39, 49)) | set(range(72, 81)) | set(range(104, 113))
cdef set _post_transition_metals = {13, 31, 49, 50, 81, 82, 83, 113, 114, 115, 116}
cdef set _lanthanides = set(range(57, 72))
cdef set _actinides = set(range(89, 104))
cdef set metal_ = _alkali_metals|_alkaline_earth_metals|_transition_metals|_post_transition_metals|_lanthanides|_actinides

cdef set _noble_gases = {2, 10, 18, 36, 54, 86, 118}
cdef set _halogens = {9, 17, 35, 53, 85, 117}

print(_default_valence[20])


cpdef bytes is_metal(int atomic_number):
    return atomic_number in metal_


cpdef int max_int(int* (*list_int)()):
    max_v = 0
    for i in list_int():
        if i > max_v:
            max_v = i

    return max_v


cdef int get_valence(int atomic_number, list[int] (*get_neigh_atomic_number)(), int (*sum_covalent_orders)()):
    if atomic_number in [6, 14]:  # C, Si
        return 4
    elif atomic_number == 8:  # O
        return 2
    elif atomic_number == 7:  # N
        if all(an != 8 for an in get_neigh_atomic_number()):
            return 3
        else:
            return max(5, 2 * len([an for an in get_neigh_atomic_number() if an == 8]) + 1)
    elif atomic_number == 15:  # P
        if all(an != 8 for an in get_neigh_atomic_number()):
            return 3
        else:
            return max(5, 2 * len([an for an in get_neigh_atomic_number() if an == 8]) + 1)
    elif atomic_number == 16:  # S
        if all(na != 8 for na in get_neigh_atomic_number()):
            return 2
        elif sum_covalent_orders() <= 4:
            return 4
        else:
            return 6
    elif atomic_number == 5:  # B
        return 3
    elif atomic_number == 1 or atomic_number in _halogens:
        return 1
    elif is_metal(atomic_number):
        return _default_valence[atomic_number]
    elif atomic_number in _noble_gases:
        return 0
    else:
        return _default_valence[atomic_number]

