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
    from openbabel import openbabel as ob, pybel as pb

    if isinstance(src, bytes):
        src = src.decode('utf-8')
    if src_type == 'path':
        return (pmol.OBMol for pmol in pb.readfile(fmt, str(src), ob_opt))

    conversion = ob.OBConversion()
    if not conversion.SetInFormat(fmt):
        raise ValueError(f"{fmt} is not a recognised Open Babel format")
    for key, value in ob_opt.items():
        if value is None:
            conversion.AddOption(key, conversion.INOPTIONS)
        else:
            conversion.AddOption(key, conversion.INOPTIONS, str(value))
    first = ob.OBMol()
    if not conversion.ReadString(first, src):
        raise OSError(f"Failed to convert '{src}' to format '{fmt}'")

    def _generator():
        obmol = first
        while True:
            yield obmol
            obmol = ob.OBMol()
            if not conversion.Read(obmol):
                return
    return _generator()


def _write_string(conversion, obmol):
    _initialize_logging()
    return conversion.WriteString(obmol)
