"""Exceptions raised by Hotpot's active SMARTS implementation."""


class SmartsSyntaxError(ValueError):
    """The SMARTS text is syntactically invalid."""


class UnsupportedSmartsError(NotImplementedError):
    """The SMARTS text is valid but uses an unsupported feature."""
