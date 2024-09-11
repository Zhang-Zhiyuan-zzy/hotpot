"""
python v3.9.0
@Project: hotpot0.5.0
@File   : utils
@Auther : Zhiyuan Zhang
@Data   : 2024/6/5
@Time   : 15:10
"""
import os
import time
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup


def download_pubchem_sdf_files(output_dir, base_url="https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/"):
    # Expand user directory
    output_dir = os.path.expanduser(output_dir)

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Fetch the webpage content
    response = requests.get(base_url)
    response.raise_for_status()  # Ensure we notice bad responses

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all .sdf.gz links
    links = []
    for link in soup.find_all('a', href=True):
        if link['href'].endswith('.sdf.gz'):
            full_url = os.path.join(base_url, link['href'])
            print(full_url)
            links.append(full_url)

    # Download the files
    for i, file_url in enumerate(links):
        print(f"Downloading {i+1}/{len(links)}:")
        file_name = os.path.join(output_dir, os.path.basename(file_url))
        download_file(file_url, file_name)

    print(f"All .sdf.gz files have been downloaded to: {output_dir}")


def download_file(url, dest_path):
    url_suffix = url.split('/')[-1]
    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        # Get the total file size from the response headers
        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as file, tqdm(
            desc=f"Download from {url_suffix}...",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            ncols=120,
        ) as bar:
            start_time = time.time()
            downloaded_size = 0

            for chunk in response.iter_content(chunk_size=262144):
                if chunk:  # Filter out keep-alive new chunks.
                    file.write(chunk)
                    chunk_size = len(chunk)
                    downloaded_size += chunk_size
                    elapsed_time = time.time() - start_time

                    # Calculate download speed
                    speed = (downloaded_size / elapsed_time) / 1024  # KB/s

                    # Update the progress bar
                    bar.set_postfix(speed=f'{speed:.2f} KB/s')
                    bar.update(chunk_size)

#
# def download_file(url, dest_path):
#     with requests.get(url, stream=True) as response:
#         response.raise_for_status()
#         with open(dest_path, 'wb') as file:
#             for chunk in response.iter_content(chunk_size=8192):
#                 file.write(chunk)


# Example usage of the function
if __name__ == "__main__":
    out_dir = r'D:\pubchem'
    download_pubchem_sdf_files(out_dir)
