"""Open Babel source iteration and serialization of task-local external values."""

_LOGGING_INITIALIZED = False


def _initialize_logging():
    global _LOGGING_INITIALIZED
    if not _LOGGING_INITIALIZED:
        from openbabel import openbabel as ob, pybel as pb

        ob_log_handler = ob.OBMessageHandler()
        ob_log_handler.SetOutputLevel(0)
        ob.obErrorLog.StopLogging()
        pb.ob.obErrorLog.StopLogging()
        _LOGGING_INITIALIZED = True


def get_ob_conversion(fmt='smi', **kwargs):
    _initialize_logging()
    from openbabel import openbabel as ob

    conv = ob.OBConversion()
    conv.SetOutFormat(fmt)

    for k, v in kwargs.items():
        if v is None:
            conv.AddOption(k, conv.OUTOPTIONS)
        else:
            conv.AddOption(k, conv.OUTOPTIONS, str(v))

    return conv


def write_obmol_to_string(obmol: "ob.OBMol", fmt='smi', **kwargs):
    return get_ob_conversion(fmt, **kwargs).WriteString(obmol)


def _read_source(src, src_type, fmt, ob_opt):
    _initialize_logging()
    from openbabel import pybel as pb

    if src_type == 'IOString':
        src = src.read()
    if src_type == 'path':
        reader = pb.readfile(fmt, str(src), ob_opt)
    else:
        reader = pb.readstring(fmt, src, ob_opt)

    def _generator():
        nonlocal reader
        if isinstance(reader, pb.Molecule):
            reader = [reader]
        for pmol in reader:
            yield pmol.OBMol
    return _generator()


def _write_string(conversion, obmol):
    _initialize_logging()
    return conversion.WriteString(obmol)
