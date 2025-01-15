_atomic_orbital = [  # Periodic
    [2],  # 1: 1s
    [2, 6],  # 2: 2s, 2p
    [2, 6],  # 3: 3s, 3p
    [2, 10, 6],  # 4: 4s, 3d, 4p
    [2, 10, 6],  # 5: 5s, 4d, 5p
    [2, 14, 10, 6],  # 6: 6s, 4f, 5d, 6p
    [2, 14, 10, 6],  # 7: 7s, 5f, 6d, 7p
    [2, 18, 14, 10, 6]  # 8: 8s, 5g, 6f, 7d, 8p
]

def calc_electron_config(_atomic_number):
    shells = _atomic_orbital
    #       s  p  d  f, g
    conf = [0, 0, 0, 0, 0]

    n = 0
    l = 0
    while _atomic_number > 0:
        if l >= len(shells[n]):
            n += 1
            l = 0
            conf = [0, 0, 0, 0, 0]

        if _atomic_number - shells[n][l] > 0:
            conf[l] = shells[n][l]
        else:
            conf[l] = _atomic_number

        _atomic_number -= shells[n][l]
        l += 1

    return n, l, conf
