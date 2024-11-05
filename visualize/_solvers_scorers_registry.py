"""
This script serves as a central registry for domain adaptation (DA) methods, scorers, and datasets.
It provides a comprehensive structure for grouping and referencing various methods and datasets used in domain adaptation tasks.
When adding new methods, scorers, or datasets, users need to update the respective dictionaries, ensuring a consistent and organized framework.

Domain Adaptation Techniques
-----------------------------

The `DA_TECHNIQUES` dictionary categorizes different domain adaptation techniques into five groups:

- NO DA: Methods that do not perform domain adaptation.
- Reweighting: Methods that reweight the source data to make it closer to the target data.
- Mapping: Methods that find a mapping between the source and target data that minimizes the distribution shift.
- Subspace: Methods that learn a subspace where the source and target data have the same distribution.
- Other: Various other methods that do not fit into the above categories.

Scorer Dictionary
-----------------

The `SCORER_DICT` dictionary maps scorer methods to their abbreviations.
These scorers are used to evaluate the performance of domain adaptation techniques.

Estimator Dictionary
--------------------

The `ESTIMATOR_DICT` dictionary maps each domain adaptation method to a corresponding estimator abbreviation.
This provides a shorthand notation for each technique.

Dataset Dictionary
------------------

The `DATASET_DICT` dictionary categorizes different datasets used in domain adaptation tasks into four groups:
- Computer Vision: Datasets commonly used for computer vision tasks.
- NLP: Datasets used for natural language processing tasks.
- Tabular: Datasets in tabular format.
- Biosignals: Datasets related to biological signals.

Adding a New Method - Scorer
----------------------------

To add a new domain adaptation method or scorer:

1. Domain Adaptation Technique: Add the method to the appropriate category in the `DA_TECHNIQUES` dictionary.
                                + Add its abbreviation to the `ESTIMATOR_DICT` dictionary.
2. Scorer: Add the new scorer method and its abbreviation to the `SCORER_DICT` dictionary.
3. Dataset: Add the new dataset to the appropriate category in the `DATASET_DICT` dictionary.

This centralized approach ensures that all domain adaptation methods, scorers, and datasets are consistently organized and easily accessible.
"""  # noqa: E501

DA_TECHNIQUES = {
    'NO DA': [
        'NO_DA_SOURCE_ONLY',
        'NO_DA_TARGET_ONLY'
    ],
    'Reweighting': [
        'gaussian_reweight',
        'KLIEP',
        'discriminator_reweight',
        'KMM',
        "TarS",
        'density_reweight',
        'nearest_neighbor_reweight',
    ],
    'Mapping': [
        'CORAL',
        'MMDSConS',
        'linear_ot_mapping',
        'entropic_ot_mapping',
        'ot_mapping',
        'class_regularizer_ot_mapping',
    ],
    'Subspace': [
        'transfer_component_analysis',
        'subspace_alignment',
        'transfer_subspace_learning',
        'joint_distribution_adaptation',
        'conditional_transferable_components',
        'transfer_joint_matching',
        'PCA',
    ],
    'Other': [
        'JDOT_SVC',
        'DASVM',
        'OTLabelProp'
    ],
}

SCORER_DICT = {
    'supervised_scorer': 'SS',
    'importance_weighted': 'IWG',
    'soft_neighborhood_density': 'SND',
    'deep_embedded_validation': 'DV',
    'circular_validation': 'CircV',
    'prediction_entropy': 'PE',
}

ESTIMATOR_DICT = {
    'CORAL': 'CORAL',
    'JDOT_SVC': 'JDOT',
    'KLIEP': 'KLIEP',
    'PCA': 'JPCA',
    'discriminator_reweight': 'Disc. RW',
    'entropic_ot_mapping': 'EntOT',
    'gaussian_reweight': 'Gauss. RW',
    'linear_ot_mapping': 'LinOT',
    'ot_mapping': 'MapOT',
    'subspace_alignment': 'SA',
    'transfer_component_analysis': 'TCA',
    'transfer_joint_matching': 'TJM',
    'transfer_subspace_learning': 'TSL',
    'joint_distribution_adaptation': 'JDA',
    'conditional_transferable_components': 'CTC',
    'class_regularizer_ot_mapping': 'ClassRegOT',
    'MMDSConS': 'MMD-LS',
    'TarS': 'MMDTarS',
    'KMM': 'KMM',
    'NO_DA_SOURCE_ONLY': 'Train Src',
    'NO_DA_TARGET_ONLY': 'Train Tgt',
    'DASVM': 'DASVM',
    'density_reweight': 'Dens. RW',
    'nearest_neighbor_reweight': 'NN RW',
    'OTLabelProp': 'OTLabelProp'
}

DEEP_ESTIMATOR_DICT = {
    'deep_no_da_source_only': 'Train Src',
    'deep_no_da_target_only': 'Train Tgt',
    'deep_coral': 'DeepCORAL',
    'deep_dan': 'DAN',
    'deep_dann': 'DANN',
    'deep_jdot': 'DeepJDOT',
    'deep_mcc': 'MCC',
}

DATASET_DICT = {
  'Computer_vision': [
    'Office31',
    'OfficeHomeResnet',
    'mnist_usps',
  ],
  'NLP': [
    '20NewsGroups',
    'AmazonReview',
  ],
  'Tabular': [
    'Mushrooms',
    'Phishing',
  ],
  'Biosignals': [
    'BCI',
  ],
}

DEEP_DATASET_DICT = {
  'deep_mnist_usps': 'MNIST/USPS',
  'deep_office31': 'Office31',
}
