"""Small public dataset API; advanced normalized storage lives in ``detail``."""

from .public.bundle import Bundle, load_bundle, write_certificate
from .public.catalog import datasets_list, load
from .public.dataset import Data, Dataset
from .public.search import BaseSearcher, DataSearcher, DatasetSearcher, SmartsSearcher


__all__ = [
    "datasets_list",
    "load",
    "Dataset",
    "Data",
    "BaseSearcher",
    "DataSearcher",
    "DatasetSearcher",
    "SmartsSearcher",
    "load_bundle",
    "Bundle",
    "write_certificate",
]


def __dir__():
    return list(__all__)


# Importing implementation submodules registers them on their parent package. Remove those
# incidental attributes so normal first-level introspection matches the documented facade.
for _module_name in ("database", "levels", "models", "properties", "public", "quality"):
    globals().pop(_module_name, None)
del _module_name
