import os
import requests
import gzip
import zipfile
from tqdm import tqdm


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


def uncompress_zip(zip_path, extract_dir):
    """
    Unzips the file at zip_path into the directory extract_dir.
    Shows a progress bar via tqdm.
    """
    # Ensure the directory exists
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Gather all items in the zip
        members = zip_ref.infolist()

        # Iterate and extract each file with a progress bar
        for file_info in tqdm(members, desc="Unzipping", unit="files"):
            zip_ref.extract(file_info, path=extract_dir)
