import os
import shutil

from hotpot.ds._paths import DATA_ROOT
from .. import _io
from .pyg_data import LnQM_Dataset


SOURCE_ARCHIVE_URL = 'https://zenodo.org/api/records/10406124/files-archive'

_data_dir = DATA_ROOT / "sources" / "lnqm"

_download_file_name = 'files-archive'
_download_file_path = _data_dir / _download_file_name

_h5file = _data_dir / 'lnqm.h5'


class LnQmDataset:
    def __init__(self):
        _data_dir.mkdir(parents=True, exist_ok=True)

        # Check whether the files complete.
        # If the file is not complete download and unzip file.
        if not (_data_dir / 'Done').exists():
            for file in _data_dir.iterdir():
                if file.is_dir():
                    shutil.rmtree(file)
                else:
                    os.remove(file)

            _io.download_file(_download_file_path, SOURCE_ARCHIVE_URL)
            _io.uncompress_zip(_download_file_path, _data_dir)
            _io.uncompress_zip(_data_dir / 'lnqm_geometries.zip', _data_dir)

            # Remove intermediate files
            os.remove(_download_file_path)
            os.remove(_data_dir / 'lnqm_geometries.zip')

            with open(_data_dir / 'Done', 'w') as f:
                f.write('Done')

    def to_pyg_dataset(self):
        return LnQM_Dataset(_h5file)
