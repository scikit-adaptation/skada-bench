""" File to preprocess AmazonReview Dataset.
- Download the dataset
- Vectorize the text using sentence_transformers
- Store the preprocessed data in a dictionary and save it in a pickle file.
- The pickle file is stored in the datasets folder.
"""

import urllib
import gzip
import tarfile
import shutil
import re
from pathlib import Path

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


def load_amazon(source, target, path=None):
    """
    Load sentiment analysis dataset giving source and target domain names
    """
    if path is None:
        path = Path(__file__).parent
    try:
        file = open(path / "kitchen.txt")
    except:
        print("Downloading sentiment analysis data files...")
        download_amazon(path)
        print(
            "Sentiment analysis data files successfully downloaded and saved in 'dataset/sa' folder"
        )
    return _get_Xy(source, target, path)


def _get_reviews_and_labels_from_txt(file):
    """
    Open text file of amazon reviews and add all reviews and
    ratings in two list.
    max_ gives the maximum number of reviews to extract.
    """
    file = open(file)

    reviews = []
    labels = []
    capture_label = False
    capture_review = False
    for line in file:
        if capture_label:
            labels.append(float(line))
        if capture_review:
            reviews.append(str(line))

        capture_label = False
        capture_review = False

        if "<rating>" in line:
            capture_label = True
        if "<review_text>" in line:
            capture_review = True
    return reviews, labels


def _decontracted(phrase):
    """
    Decontract english common contraction
    """
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can't", "can not", phrase)
    phrase = re.sub(r"n\'t", "not", phrase)
    phrase = re.sub(r"\'re", "are", phrase)
    phrase = re.sub(r"\'s", "is", phrase)
    phrase = re.sub(r"\'d", "would", phrase)
    phrase = re.sub(r"\'ll", "will", phrase)
    phrase = re.sub(r"\'t", "not", phrase)
    phrase = re.sub(r"\'ve", "have", phrase)
    phrase = re.sub(r"\'m", "am", phrase)
    return phrase


def _preprocess_review(reviews):
    """
    Preprocess text in reviews list
    """

    # Download stopwords if not already downloaded
    try:
        stop = set(stopwords.words("english"))
    except LookupError:
        import nltk

        nltk.download("stopwords")
        stop = set(stopwords.words("english"))

    snow = SnowballStemmer("english")

    preprocessed_reviews = []

    for sentence in reviews:
        sentence = re.sub(r"http\S+", " ", sentence)
        cleanr = re.compile("<.*?>")
        sentence = re.sub(cleanr, " ", sentence)
        sentence = _decontracted(sentence)
        sentence = re.sub("\S\*\d\S*", " ", sentence)
        sentence = re.sub("[^A-Za-z]+", " ", sentence)
        sentence = re.sub(r'[?|!|\'|"|#]', r" ", sentence)
        sentence = re.sub(r"[.|,|)|(|\|/]", r" ", sentence)
        sentence = "  ".join(
            snow.stem(e.lower()) for e in sentence.split() if e.lower() not in stop
        )
        preprocessed_reviews.append(sentence.strip())

    return preprocessed_reviews


def get_reviews(domain, path):
    """
    Return preprocessed reviews and labels
    """
    reviews, labels = _get_reviews_and_labels_from_txt(path / f"{domain}.txt")
    reviews = _preprocess_review(reviews)
    return reviews, labels


def download_amazon(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

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
        path / "domain_sentiment_data/sorted_data_acl/kitchen_&_housewares/unlabeled.review",
        path / "kitchen.txt",
    )

    (path / "book.unlabeled.gz").unlink()
    (path / "domain_sentiment_data.tar.gz").unlink()
    shutil.rmtree(path / "domain_sentiment_data")


if __name__ == "__main__":
    PATH = Path("data/amazon_review")
    DOMAINS = ["books", "dvd", "electronics", "kitchen"]

    # Download data
    download_amazon(PATH)

    # Get (X, y) for each domain
    for domain in DOMAINS:
        X, y = get_reviews(domain, PATH)

