"""Download helpers for the internal tmQM source reader."""
import os

from hotpot.ds._paths import DATA_ROOT
from .._io import download_file, uncompress_all_gz

_data_dir = DATA_ROOT / "sources" / "tmqm"


__all__ = [
    'urls',
    'download_file',
    'download_files',
    'uncompress_all_gz',
]

# List of URLs to download
urls = {
    'xyz': 'https://www.uiocompcat.info/img/g177830444-o63018611.dat?dl=2&tk=qPEqUMZJfPTiH-_ME-ESvjaHKpUSCfGEnFtzVI7SnW0=',
    'csv': 'https://www.uiocompcat.info/img/g453053122-o63018611.dat?dl=2&tk=HcAuiBMEDJPlJnx9xj9OJdo4YQ8M07qGik-6wz8r7QA=',
    'q': 'https://www.uiocompcat.info/img/g330963833-o63018611.dat?dl=2&tk=Ik2psodHxr4BW0spTmXbLfWILuEgJFUe3_4GhZSeq6U=',
    'BO': 'https://www.uiocompcat.info/img/g525088609-o63018611.dat?dl=2&tk=V-tSIEKy98M0FtrC9Nx-tip0GsUD6NGTmA6oBbbNSsg='
}


def download_files():
    # Downloading each file from the list
    _data_dir.mkdir(parents=True, exist_ok=True)
    for ext, url in urls.items():
        file_path = _data_dir / f'tmQm.{ext}.gz'
        download_file(file_path, url)
        uncompress_all_gz(file_path)
        os.remove(file_path)  # Remove the gz file
if __name__ == '__main__':
    download_files()
