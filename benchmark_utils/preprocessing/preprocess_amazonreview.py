""" File to preprocess AmazonReview Dataset.
- Download the dataset
- Vectorize the text using sentence_transformers
- Store the preprocessed data in a dictionary and save it in a pickle file.
- The pickle file is stored in the datasets folder.
"""

import urllib.request
import gzip
import tarfile
import shutil
from pathlib import Path
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


def download_amazon(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Check if the file is already downloaded
    # If not, download it
    if not (path / "kitchen.txt").exists():
        urllib.request.urlretrieve(
            "https://www.cs.jhu.edu/~mdredze/datasets/"
            "sentiment/domain_sentiment_data.tar.gz",
            path / "domain_sentiment_data.tar.gz",
        )
        urllib.request.urlretrieve(
            "https://www.cs.jhu.edu/~mdredze/datasets/"
            "sentiment/book.unlabeled.gz",
            path / "book.unlabeled.gz",
        )
        tar = tarfile.open(path / "domain_sentiment_data.tar.gz", "r:gz")
        tar.extractall(path / "domain_sentiment_data")
        tar.close()
        with gzip.open(path / "book.unlabeled.gz", "rb") as f_in:
            with open(path / "books.txt", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        shutil.move(
            path / "domain_sentiment_data/sorted_data_acl/"
            "dvd/unlabeled.review",
            path / "dvd.txt",
        )
        shutil.move(
            path / "domain_sentiment_data/sorted_data_acl/"
            "electronics/unlabeled.review",
            path / "electronics.txt",
        )
        shutil.move(
            path / "domain_sentiment_data/sorted_data_acl/"
            "kitchen_&_housewares/unlabeled.review",
            path / "kitchen.txt",
        )

        (path / "book.unlabeled.gz").unlink()
        (path / "domain_sentiment_data.tar.gz").unlink()
        shutil.rmtree(path / "domain_sentiment_data")


def get_reviews(domain, path):
    """
    Open text file of amazon reviews and add all reviews and
    ratings in two list.
    max_ gives the maximum number of reviews to extract.
    """
    file = open(path / f"{domain}.txt")

    reviews = []
    labels = []
    capture_label = False
    capture_review = False
    for line in file:
        if capture_label:
            labels.append(float(line))
        if capture_review:
            reviews.append(str(line).strip())

        capture_label = False
        capture_review = False

        if "<rating>" in line:
            capture_label = True
        if "<review_text>" in line:
            capture_review = True
    return reviews, np.array(labels)


def preprocess_labels(labels):
    """
    Preprocess the labels to be in {0, 1, 2, 3}.
    """
    # Check if the labels are in [1, 2, 4, 5]
    assert np.all(np.isin(labels, [1, 2, 4, 5]))
    # Use Label
    labels[labels == 1] = 0
    labels[labels == 2] = 1
    labels[labels == 4] = 2
    labels[labels == 5] = 3
    return labels


if __name__ == "__main__":
    PATH = Path("data")
    PATH_RAW = PATH / "amazon_review_raw"
    DOMAINS = ["books", "dvd", "electronics", "kitchen"]

    N_COMPONENTS = 50
    MODEL_NAME = "BAAI/bge-large-en-v1.5"
    BATCH_SIZE = 512
    # MODEL_NAME = "BAAI/bge-small-en-v1.5"
    # BATCH_SIZE = 1024
    DEVICE = "cuda"

    # Download data
    print("Downloading data...")
    download_amazon(PATH_RAW)

    # Get (X, y) for each domain
    print("Getting (X, y) for each domain...")
    data_raw = dict()
    for domain in DOMAINS:
        X, y = get_reviews(domain, PATH_RAW)
        data_raw[domain] = {'X': X, 'y': y}
    for k, v in data_raw.items():
        print(f"{k}: {len(v['X'])} reviews, {len(v['y'])} labels")

    # Sentence Transformers
    print(
        "Encoding text data using Sentence Transformers and merging labels..."
    )
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    preprocessed_data = dict()
    for k in data_raw.keys():
        print(f"Processing {k}...")
        preprocessed_data[k] = dict()
        preprocessed_data[k]['X'] = model.encode(
            data_raw[k]['X'], batch_size=BATCH_SIZE, show_progress_bar=True
        )
        preprocessed_data[k]['y'] = preprocess_labels(data_raw[k]['y'])

    # Apply PCA to reduce the dimensionality of the embeddings
    print("Applying PCA...")
    X_total = np.concatenate([v['X'] for v in preprocessed_data.values()])
    pca = PCA(n_components=N_COMPONENTS).fit(X_total)
    for k in data_raw.keys():
        preprocessed_data[k]['X'] = pca.transform(preprocessed_data[k]['X'])

    # Save the preprocessed data
    print("Saving preprocessed data...")
    # Change X key to sentence_transformers
    for k in preprocessed_data.keys():
        preprocessed_data[k]['sentence_transformers'] = (
            preprocessed_data[k].pop('X')
        )
    PATH.mkdir(exist_ok=True)
    with open(PATH / "amazon_review_preprocessed.pkl", "wb") as f:
        pickle.dump(preprocessed_data, f)
