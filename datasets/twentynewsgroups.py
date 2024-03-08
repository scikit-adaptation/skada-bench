from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer

    from skada.utils import source_target_merge


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "20NewsGroups"

    install_cmd = 'conda'

    requirements = ['scikit-learn']

    references = [
        "Dai W., Yang Q., Xue G., and Yu Y, "
        "'Boosting for transfer learning' "
        "ICML, (2007)."
    ]

    parameters = {
        "source_target": [
            ("rec", "talk"),
            ("rec", "sci"),
            ("talk", "sci"),
            ("talk", "rec"),
            ("sci", "rec"),
            ("sci", "talk"),
        ],
        "max_features": [5000]
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Set download_if_missing to True if not downloaded yet
        data = fetch_20newsgroups(download_if_missing=True, subset="all")

        vectorizer = TfidfVectorizer(stop_words="english",
                                     analyzer="word",
                                     min_df=5,
                                     max_df=0.1,
                                     max_features=self.max_features)

        X = vectorizer.fit_transform(data.data)

        source = self.source_target[0]
        target = self.source_target[1]

        source_dict = {
            "rec": ['rec.autos', 'rec.motorcycles'],
            "talk": ['talk.politics.guns', 'talk.politics.misc'],
            "sci": ['sci.crypt', 'sci.electronics']
        }

        target_dict = {
            "rec": ['rec.sport.baseball', 'rec.sport.hockey'],
            "talk": ['talk.religion.misc', 'talk.politics.mideast'],
            "sci": ['sci.med', 'sci.space']
        }

        source_index = np.isin(
            data.target,
            [
                data.target_names.index(s)
                for s in source_dict[source]+source_dict[target]
            ]
        )
        target_index = np.isin(
            data.target,
            [
                data.target_names.index(s)
                for s in target_dict[source]+target_dict[target]
            ]
        )
        positive_index = [
            data.target_names.index(s)
            for s in target_dict[source]+source_dict[source]
        ]

        X_source = X[source_index].toarray()
        X_target = X[target_index].toarray()

        y_source = np.isin(data.target[source_index],
                           positive_index).astype(float)
        y_target = np.isin(data.target[target_index],
                           positive_index).astype(float)

        X, y, sample_domain = source_target_merge(
            X_source, X_target, y_source, y_target)

        return dict(
            X=X,
            y=y,
            sample_domain=sample_domain
        )
