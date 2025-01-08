"""
python v3.9.0
@Project: hotpot
@File   : download
@Auther : Zhiyuan Zhang
@Data   : 2024/12/24
@Time   : 15:33
"""
"""
python v3.9.0
@Project: hotpot
@File   : tmqm
@Auther : Zhiyuan Zhang
@Data   : 2024/12/24
@Time   : 14:41

Notes:
"""
import os
from os import path
from os.path import join as opj
import gzip
import shutil
import requests

from tqdm import tqdm

_module_root = path.dirname(__file__)
_data_dir = opj(_module_root, 'data')


__all__ = [
    'urls',
    'download_file',
    'download_files'
]

# List of URLs to download
urls = {
    'xyz': 'https://www.uiocompcat.info/img/g177830444-o63018611.dat?dl=2&tk=qPEqUMZJfPTiH-_ME-ESvjaHKpUSCfGEnFtzVI7SnW0=',
    'csv': 'https://www.uiocompcat.info/img/g453053122-o63018611.dat?dl=2&tk=HcAuiBMEDJPlJnx9xj9OJdo4YQ8M07qGik-6wz8r7QA=',
    'q': 'https://www.uiocompcat.info/img/g330963833-o63018611.dat?dl=2&tk=Ik2psodHxr4BW0spTmXbLfWILuEgJFUe3_4GhZSeq6U=',
    'BO': 'https://www.uiocompcat.info/img/g525088609-o63018611.dat?dl=2&tk=V-tSIEKy98M0FtrC9Nx-tip0GsUD6NGTmA6oBbbNSsg='
}


# Function to download files
def download_file(fp, _url):
    response = requests.get(_url)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(fp, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloaded: {fp} ...") as bar:
                for data in response.iter_content(chunk_size=1024):  # Download in chunks
                    f.write(data)
                    bar.update(len(data))  # Update the progress bar
        # print(f"Downloaded: {fp}")
    else:
        print(f"Failed to download: {_url} (Status Code: {response.status_code})")


def download_files():
    # Downloading each file from the list
    for ext, url in urls.items():
        file_path = opj(_data_dir, f'tmQm.{ext}.gz')
        download_file(file_path, url)
        uncompress_all_gz(file_path)
        os.remove(file_path)  # Remove the gz file


def uncompress_all_gz(file_path):
    output_path = os.path.join(file_path[:-3])  # Remove .gz extension
    with gzip.open(file_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
        # Get the total size of the compressed file for the progress bar
        total_size = os.path.getsize(file_path)

        # Create a progress bar
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f'Uncompressing {file_path}') as bar:
            # Copy the file in chunks
            chunk_size = 1024  # 1 KB
            while True:
                data = f_in.read(chunk_size)
                if not data:
                    break
                f_out.write(data)
                bar.update(len(data))  # Update the progress bar

    # print(f'Uncompressed: {file_path} to {output_path}')

if __name__ == '__main__':
    download_files()
