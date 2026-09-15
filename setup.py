"""Compatibility shim for legacy setuptools frontends.

Project metadata and dependency declarations live exclusively in
``pyproject.toml``.
"""

from setuptools import setup


if __name__ == "__main__":
    setup()
