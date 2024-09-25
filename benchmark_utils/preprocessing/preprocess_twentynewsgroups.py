""" File to preprocess 20NewsGroups Dataset.
- Download the dataset
- Vectorize the text data using MinHashEncoder and sentence_transformers
- Store the preprocessed data in a dictionary and save it in a pickle file.
- The pickle file is stored in the datasets folder.
"""

from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from skrub import MinHashEncoder


if __name__ == "__main__":
    # MinHashEncoder hyperparameters
    N_COMPONENTS = 50
    NGRAM_RANGE = (2, 10)

    # Sentence Transformers hyperparameters
    MODEL_NAME = "BAAI/bge-large-en-v1.5"
    BATCH_SIZE = 512
    # MODEL_NAME = "BAAI/bge-small-en-v1.5"
    # BATCH_SIZE = 1024
    DEVICE = "cuda"

    # Set download_if_missing to True if not downloaded yet
    data = fetch_20newsgroups(download_if_missing=True, subset="all")

    # MinHashEncoder
    print("Encoding text data using MinHashEncoder...")
    vectorizer = MinHashEncoder(
        n_components=N_COMPONENTS, ngram_range=NGRAM_RANGE, n_jobs=-1
    )
    X_min_hash = vectorizer.fit_transform([[d] for d in data.data])

    # Sentence Transformers
    print("Encoding text data using Sentence Transformers...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    X_sentence_transformers = model.encode(
        data.data, batch_size=BATCH_SIZE, show_progress_bar=True
    )
    X_sentence_transformers = X_sentence_transformers.astype("float64")

    # Apply PCA to reduce the dimensionality of the embeddings
    print("Applying PCA...")
    pca = PCA(n_components=N_COMPONENTS)
    X_sentence_transformers = pca.fit_transform(X_sentence_transformers)

    # Save the preprocessed data
    print("Saving preprocessed data...")
    preprocessed_data = {
        "raw_data": data.data,
        "min_hash": X_min_hash,
        "sentence_transformers": X_sentence_transformers,
    }

    # Save the preprocessed data in a pickle file
    path = Path('data')
    path.mkdir(exist_ok=True)
    with open(path / "20newsgroups_preprocessed.pkl", "wb") as f:
        pickle.dump(preprocessed_data, f)
