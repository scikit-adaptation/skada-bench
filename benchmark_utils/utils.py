# Utils file for useful functions
import os
import time
import requests
import zipfile
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


def _download_file_with_progress(url, filename):
    """
    Download a file from a given URL with a progress bar.
    """
    # Streaming download because the download
    # is sometimes so long that it seems broken
    response = requests.get(url, stream=True, timeout=5)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 16384 #16 Kilobyte
    downloaded_size = 0
    start_time = time.time()

    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            size = file.write(data)
            downloaded_size += size
            percent = int(50 * downloaded_size / total_size_in_bytes)
            elapsed_time = time.time() - start_time
            avg_speed = downloaded_size / (elapsed_time * 1024 * 1024)  # MB/s
            print(f"\rDownloading: [{'#' * percent}{' ' * (50-percent)}] {percent*2}% ({avg_speed:.2f} MB/s)", end='', flush=True)

    print("\nDownload completed.")


def download_and_extract_zipfile(url_dataset, path_dataset, path_extract):
    """
    Download a zip file from a URL and extract its contents.
    """
    # Check if the data is available
    # If not, download them
    path_dataset = Path(path_dataset)
    path_extract = Path(path_extract)

    if not path_extract.exists():
        print(f"Downloading dataset from {url_dataset}")
        try:
            _download_file_with_progress(url_dataset, path_dataset)
        except Exception as e:
            print(f"\nError downloading the dataset: {e}")
            return

        # Load data
        try:
            with zipfile.ZipFile(path_dataset, "r") as zip_ref:
                print(f"Extracting files to {path_extract}")
                zip_ref.extractall(path_extract)
            print("Extraction complete")
        except zipfile.BadZipFile:
            print("The file is still not a valid zip file after re-downloading. Please check the URL or your internet connection.")
            return
        except Exception as e:
            print(f"Error extracting the zip file: {e}")
            return

        # Remove the zip file after successful extraction
        os.remove(path_dataset)
        print(f"Removed zip file: {path_dataset}")

        print("Process completed successfully")


class ImageDataset(Dataset):
    """
    Custom dataset class to load images from a specific domain.
    """
    def __init__(self, dataset_dir, domain_select, transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.image_paths = []
        for image_path in self.dataset_dir.rglob('*.jpg'):
            domain = image_path.parent.parent.name
            if domain.lower() == domain_select:
                self.image_paths.append(image_path)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Extract label and domain from the image path
        label = image_path.parent.name
        domain = image_path.parent.parent.name

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        return image, label