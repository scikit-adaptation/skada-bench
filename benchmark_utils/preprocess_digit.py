""" File to preprocess Digit datasets
- Download the datasets
- preprocess and flatten the images
- Store the preprocessed data in a dictionary and save it in a pickle file.
- The pickle file is stored in the datasets folder.
"""

import numpy as np
import os
from pathlib import Path
import pickle
from sklearn.decomposition import PCA
import torch
import torchvision
from torchvision.datasets import MNIST, SVHN, USPS


if __name__ == "__main__":
    DATASETS = ['MNIST', 'SVHN', 'USPS']
    RANDOM_STATE = 27
    BATCH_SIZE = 2048
    N_COMPONENTS = 50
    N_JOBS = os.cpu_count()

    preprocessed_data = dict()
    for dataset_name in DATASETS:
        print(f"Preprocessing {dataset_name} dataset")

        dataset_name = dataset_name.lower()

        if dataset_name == 'mnist':
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Pad(2),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ])
            dataset = MNIST(
                root='./data/MNIST',
                download=True,
                train=True,
                transform=transform
            )
        elif dataset_name == 'svhn':
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ])
            dataset = SVHN(
                root='./data/SVHN',
                download=True,
                split='train',
                transform=transform
            )
        elif dataset_name == 'usps':
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Pad(8),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ])
            dataset = USPS(
                root='./data/USPS',
                download=True,
                train=True,
                transform=transform
            )
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")

        # Vectorize the images
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=N_JOBS
        )
        embeddings = list()
        with torch.no_grad():
            for images, _ in dataloader:
                embeddings.append(images)
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = embeddings.cpu().numpy()
        embeddings = embeddings.reshape(embeddings.shape[0], -1)

        # Save the preprocessed data
        preprocessed_data[dataset_name] = {
            'X': embeddings,
            'y': np.concatenate([y for _, y in dataloader])
        }

    # Apply PCA to reduce the dimensionality of the embeddings
    print("Applying PCA...")
    full_X = torch.cat(
        [
            torch.tensor(
                preprocessed_data[dataset_name.lower()]['X']
            ) for dataset_name in DATASETS
        ],
        dim=0
    )
    pca = PCA(
        n_components=N_COMPONENTS,
        whiten=True,
        random_state=RANDOM_STATE
    )
    pca.fit(full_X)
    for dataset_name in DATASETS:
        preprocessed_data[dataset_name.lower()]['X'] = pca.transform(
            preprocessed_data[dataset_name.lower()]['X']
        )

    # Save the preprocessed data in a pickle file
    print("Saving the preprocessed data...")
    path = Path('data')
    path.mkdir(exist_ok=True)
    with open(path / 'digit.pkl', 'wb') as f:
        pickle.dump(preprocessed_data, f)
