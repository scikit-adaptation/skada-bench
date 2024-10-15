from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import pandas as pd
    import numpy as np
    import urllib.request
    from pathlib import Path
    from scipy.io.arff import loadarff
    from skada.utils import source_target_merge


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Phishing"

    references = [
        "R. Mohammad, F. Thabtah, L. Mccluskey,"
        "'An assessment of features related to phishing websites using an "
        "automated technique' International Conference for Internet "
        "Technology and Secured Transactions (2012)."
    ]

    parameters = {
        "source_target": [
            ("ip_adress", "no_ip_adress"), ("no_ip_adress", "ip_adress")
        ],
    }


    path_data = "data/phishing.arff"
    url_data = "https://figshare.com/ndownloader/files/47541842"

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        columns = [
            "having-IP-adress",
            "URL-length",
            "shortining-service",
            "having-at-symbol",
            "double-slash-redirecting",
            "prefix-suffix",
            "having-sub-domain",
            "SSL-final-state",
            "domain-registration-length",
            "favicon",
            "port",
            "https-token",
            "request-URL",
            "URL-of-anchor",
            "links-in-tags",
            "sfh",
            "submitting-to-email",
            "abnormal-url",
            "redirect",
            "on-mouseover",
            "right-click",
            "pop-up-window",
            "Iframe",
            "age-of-domain",
            "dns-record",
            "web_traffic",
            "page-rank",
            "google-index",
            "links-pointing-to-page",
            "stats-report",
            "target",
        ]

        # Check if data are available
        # If not, download them
        path_data = Path(self.path_data)
        if not path_data.exists():
            urllib.request.urlretrieve(self.url_data, path_data)

        # Load data
        raw_data = loadarff(path_data)
        data = pd.DataFrame(raw_data[0])
        data.columns = columns
        data = data.astype(int)
        X = data.drop(["target"], axis=1)
        y = data["target"]

        domain_dict = {
            "no_ip_adress": -1,
            "ip_adress": 1,
        }

        source = self.source_target[0]
        target = self.source_target[1]

        X_source = X.loc[X["having-IP-adress"] == domain_dict[source]]
        y_source = y.loc[X_source.index].values
        X_target = X.loc[X["having-IP-adress"] == domain_dict[target]]
        y_target = y.loc[X_target.index].values

        X_source = X_source.to_numpy()
        X_target = X_target.to_numpy()

        X, y, sample_domain = source_target_merge(
            X_source, X_target, y_source, y_target
        )

        # Mapping from [-1, 1] to [0, 1]
        mapping = {-1: 0, 1: 1}

        # Applying the mapping function
        y = np.array([mapping[val] for val in y])

        return dict(X=X, y=y, sample_domain=sample_domain)
