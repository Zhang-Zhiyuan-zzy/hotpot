"""Version-selected public force-field API."""

import sys


if sys.version_info[:2] == (3, 9):
    from .ff39 import *
    from .ff39 import __all__ as __all__
else:
    from .ff import *
    from .ff import __all__ as __all__
