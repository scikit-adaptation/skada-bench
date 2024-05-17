from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada.utils import source_target_merge
    import numpy as np

    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.pipeline import make_pipeline

    from braindecode.datasets import MOABBDataset
    from braindecode.preprocessing import (
        exponential_moving_standardize,
        preprocess,
        Preprocessor,
    )
    from numpy import multiply
    from braindecode.preprocessing import create_windows_from_events


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "BCI"

    install_cmd = 'conda'

    requirements = ['braindecode', 'moabb', 'pyriemann']

    parameters = {
        'subject_id': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        dataset = MOABBDataset(
            dataset_name="BNCI2014001", subject_ids=[self.subject_id]
        )
        low_cut_hz = 4.0  # low cut frequency for filtering
        high_cut_hz = 40.0  # high cut frequency for filtering
        # Parameters for exponential moving standardization
        factor_new = 1e-3
        init_block_size = 1000
        # Factor to convert from V to uV
        factor = 1e6

        preprocessors = [
            Preprocessor("pick_types", eeg=True, meg=False, stim=False),
            Preprocessor(lambda data: multiply(data, factor)),
            Preprocessor(
                "filter", l_freq=low_cut_hz, h_freq=high_cut_hz
            ),
            Preprocessor(
                exponential_moving_standardize,
                factor_new=factor_new,
                init_block_size=init_block_size,
            ),
        ]

        # Transform the data
        preprocess(dataset, preprocessors)

        trial_start_offset_seconds = -0.5
        # Extract sampling frequency, check that they are same in all datasets
        sfreq = dataset.datasets[0].raw.info["sfreq"]

        assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])
        # Calculate the trial start offset in samples.
        trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

        window_size_samples = None
        window_stride_samples = None

        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=0,
            preload=False,
            window_size_samples=window_size_samples,
            window_stride_samples=window_stride_samples,
        )

        splitted = windows_dataset.split("session")
        sessions = list(splitted.keys())

        X = []
        y = []
        sess_source = sessions[0]
        n_runs = len(splitted[sess_source].datasets)
        x = []
        y = []
        for run in range(n_runs):
            x += [sample[0] for sample in splitted[sess_source].datasets[run]]
            y += [sample[1] for sample in splitted[sess_source].datasets[run]]
        X_source = np.array(x)
        y_source = np.array(y)

        sess_target = sessions[1]
        n_runs = len(splitted[sess_target].datasets)
        x = []
        y = []
        for run in range(n_runs):
            x += [sample[0] for sample in splitted[sess_target].datasets[run]]
            y += [sample[1] for sample in splitted[sess_target].datasets[run]]
        X_target = np.array(x)
        y_target = np.array(y)

        X, y, sample_domain = source_target_merge(
                    X_source, X_target, y_source, y_target)

        ts_projector = make_pipeline(
            Covariances(estimator="oas"), TangentSpace()
        )
        X = ts_projector.fit_transform(X)
        return dict(
            X=X,
            y=y,
            sample_domain=sample_domain
        )
