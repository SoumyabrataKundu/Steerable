import requests
import zipfile
import os

def download_and_unzip(url: str, download_path: str):
    """
    Download a ZIP from `url` → save to `download_path` → extract into `extract_dir`.
    """
    # Create Directories
    os.makedirs(download_path, exist_ok=True)

    # Download ZIP (streamed)
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(download_path+'.zip', "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded to {download_path+'.zip'}.")
    
    # Extract
    with zipfile.ZipFile(download_path+'.zip', "r") as z:
        z.extractall(download_path)
    print(f"Extarcted to {download_path}.")
    
    # Delete zipped file
    os.remove(download_path+'.zip')