from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import requests
    import numpy as np

    from skada.utils import source_target_merge


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    """
    Labels_id:
        - scale: 6 values linearly spaced in [0.5, 1]
        - orient: 40 values in [0, 2 pi]
        - pos_x: 32 values in [0, 1]
        - pos_y: 32 values in [0, 1]
    """
    # Name to select the dataset in the CLI and to display the results.
    name = "dSprites"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'n_samples_source': [3000],
        'n_samples_target': [3000],
        'target_id': [0],
        'labels_id': ["scale"],
        'random_state': [27],
    }

    def get_data(self):
        rng = np.random.RandomState(self.random_state)

        url = ("https://github.com/deepmind/dsprites-dataset/raw/"
               "master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")

        local_file = 'local_file.npz'

        request = requests.get(url)
        with open(local_file, 'wb') as file:
            file.write(request.content)
        data = np.load(local_file)

        dict_labels_id = dict(scale=2, orient=3, pos_x=4, pos_y=5)

        X = data["imgs"]
        y = data["latents_values"][:, dict_labels_id[self.labels_id]]

        src_index = np.argwhere(data["latents_classes"][:, 1] !=
                                self.target_id).ravel()
        indices = rng.choice(src_index, self.n_samples_source, replace=False)
        X_source = X[indices].reshape(-1, 64*64)
        y_source = y[indices]

        tgt_index = np.argwhere(data["latents_classes"][:, 1] ==
                                self.target_id).ravel()
        indices = rng.choice(tgt_index, self.n_samples_source, replace=False)
        X_target = X[indices].reshape(-1, 64*64)
        y_target = y[indices]

        X, y, sample_domain = source_target_merge(
            X_source, X_target, y_source, y_target)

        return dict(
            X=X,
            y=y,
            sample_domain=sample_domain
        )
