from benchopt import BaseDataset, safe_import_context
with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from skada.utils import source_target_merge
    import torchvision.transforms as transforms
    import os
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import torch
    import requests
    from pathlib import Path
    import zipfile
    import time



# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "OfficeHome"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'source_target': [
            ('art', 'clipart'),
            ('art', 'product'),
            ('art', 'realworld'),
            ('clipart', 'art'),
            ('clipart', 'product'),
            ('clipart', 'realworld'),
            ('product', 'art'),
            ('product', 'clipart'),
            ('product', 'realworld'),
            ('realworld', 'art'),
            ('realworld', 'clipart'),
            ('realworld', 'product'),
        ],
    }

    path_dataset = "data/OfficeHome.zip"
    path_extract = "data/OfficeHome/"
    url_dataset = "https://wjdcloud.blob.core.windows.net/dataset/OfficeHome.zip"

    def _download_file_with_progress(self, url, filename):
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

    def _download_and_extract_officehome(self):
        # Check if the data is available
        # If not, download them
        path_dataset = Path(self.path_dataset)
        path_extract = Path(self.path_extract)

        if not path_extract.exists():
            print(f"Downloading dataset from {self.url_dataset}")
            try:
                self._download_file_with_progress(self.url_dataset, path_dataset)
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
    

    def _get_dataset(self, domain_select):
        # Define transformations to preprocess the images
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Custom dataset class to load images
        class ImageDataset(Dataset):
            def __init__(self, dataset_dir, domain_select, transform=None):
                self.dataset_dir = dataset_dir
                self.image_paths = []
                for root, dirs, files in os.walk(dataset_dir):
                    for file in files:
                        if file.endswith(".jpg"):
                            image_path = os.path.join(root, file)
                            domain = os.path.basename(os.path.dirname(os.path.dirname(image_path)))

                            if np.char.lower(domain) == domain_select:
                                self.image_paths.append(os.path.join(root, file))

                self.transform = transform

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                image_path = self.image_paths[idx]

                # Extract label and domain from the image path
                label = os.path.basename(os.path.dirname(image_path))
                domain = os.path.basename(os.path.dirname(os.path.dirname(image_path)))

                image = Image.open(image_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                
                return image, label

        # Create a DataLoader for the dataset
        dataset = ImageDataset(self.path_extract, transform=preprocess, domain_select=domain_select)
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        # TODO: set batch_size=len(dataset), rn its 10k for testing purposes ONLY
        #dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

        images, labels = next(iter(dataloader))
        images = images.numpy()
        labels = np.array(labels)

        # Sklearn doesnt accept 4D arrays, thus we need to convert our input
        # from (n, c, w, h) to (n, c * w * h)
        images = images.reshape((images.shape[0], -1))

        return images, labels

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.
        self._download_and_extract_officehome()

        source, target = self.source_target
        X_source, y_source = self._get_dataset(source)
        X_target, y_target = self._get_dataset(target)

        # XGBoost only supports labels in [0, num_classes-1]
        le = LabelEncoder()
        le.fit(np.concatenate([y_source, y_target]))
        y_source = le.transform(y_source)
        y_target = le.transform(y_target)

        X, y, sample_domain = source_target_merge(
            X_source, X_target, y_source, y_target)

        # Mps device doesnt accept float64
        # TODO: Check with cuda devices
        # X = np.float32(X)
        # y = np.float32(y)
        # sample_domain = np.float32(sample_domain)

        return dict(
            X=X,
            y=y,
            sample_domain=sample_domain,
        )
