"""Geometry API migration scaffold.

The legacy implementation is re-exported temporarily while the factual
geometry modules replace it in small, independently testable steps.
"""

from ._legacy import *
from ._legacy import __all__
