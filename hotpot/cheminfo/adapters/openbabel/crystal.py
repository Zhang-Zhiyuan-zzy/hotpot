"""Fresh Open Babel cell projection from six cell parameters."""


def to_obcell(crystal) -> "ob.OBUnitCell":
    from ._runtime import ob

    obcell = ob.OBUnitCell()
    obcell.SetData(crystal.a, crystal.b, crystal.c, crystal.alpha, crystal.beta, crystal.gamma)
    return obcell
