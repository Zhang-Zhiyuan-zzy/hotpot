"""Concise public API for the one current Hotpot dataset implementation."""

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


# Importing the implementation package registers it on its parent.  Keep normal
# first-level introspection aligned with the deliberately small public facade.
globals().pop("public", None)
