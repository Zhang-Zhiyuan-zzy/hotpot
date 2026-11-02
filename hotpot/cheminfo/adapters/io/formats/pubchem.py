"""PubChem name lookup and external source extraction."""


def _read_source(src):
    from ....pubchem import pubchem_service
    from . import openbabel

    openbabel._initialize_logging()
    from openbabel import pybel as pb

    smi = pubchem_service.name_to_smi(src)
    reader = [pb.readstring('smi', smi)]

    def _generator():
        for pmol in reader:
            yield pmol.OBMol
    return _generator()
