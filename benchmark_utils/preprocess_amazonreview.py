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
import numpy as np


def download_amazon(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Check if the file is already downloaded
    # If not, download it
    if not (path / "kitchen.txt").exists():
        urllib.request.urlretrieve(
            "https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz",
            path / "domain_sentiment_data.tar.gz",
        )
        urllib.request.urlretrieve(
            "https://www.cs.jhu.edu/~mdredze/datasets/sentiment/book.unlabeled.gz",
            path / "book.unlabeled.gz",
        )
        tar = tarfile.open(path / "domain_sentiment_data.tar.gz", "r:gz")
        tar.extractall(path / "domain_sentiment_data")
        tar.close()
        with gzip.open(path / "book.unlabeled.gz", "rb") as f_in:
            with open(path / "books.txt", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        shutil.move(
            path / "domain_sentiment_data/sorted_data_acl/dvd/unlabeled.review",
            path / "dvd.txt",
        )
        shutil.move(
            path / "domain_sentiment_data/sorted_data_acl/electronics/unlabeled.review",
            path / "electronics.txt",
        )
        shutil.move(
            path
            / "domain_sentiment_data/sorted_data_acl/kitchen_&_housewares/unlabeled.review",
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
    return reviews, labels


if __name__ == "__main__":
    PATH = Path("data")
    DOMAINS = ["books", "dvd", "electronics", "kitchen"]

    # Download data
    print("Downloading data...")
    download_amazon(PATH / "amazon_review_raw")

    # Get (X, y) for each domain
    print("Getting (X, y) for each domain...")
    data_raw = dict()
    for domain in DOMAINS:
        X, y = get_reviews(domain, PATH / "amazon_review_raw")
        data_raw[domain] = {"X": X, "y": y}
    for k, v in data_raw.items():
        print(f"{k}: {len(v['X'])} reviews, {len(v['y'])} labels")
