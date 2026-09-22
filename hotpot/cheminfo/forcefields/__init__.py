"""Version-selected public force-field API."""

import sys


if sys.version_info[:2] == (3, 9):
    from .ff39 import *  # noqa: F403
    from .ff39 import __all__ as __all__
else:
    from .ff import *  # noqa: F403
    from .ff import __all__ as __all__
