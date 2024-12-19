| estimator   |   MNIST/USPS |   Office31 |   OfficeHome |   BCI | scorer                   |   rank |
|:------------|-------------:|-----------:|-------------:|------:|:-------------------------|-------:|
| Train Src   |         0.85 |       0.77 |         0.58 |  0.54 | NA                       |   6.19 |
| Train Tgt   |         0.98 |       0.96 |         0.83 |  0.56 | NA                       |   2.07 |
| DeepCORAL   |         0.93 |       0.77 |         0.59 |  0.54 | mix_val_intra            |   3.29 |
| DAN         |         0.86 |       0.75 |         0.56 |  0.53 | importance_weighted      |   4.76 |
| DANN        |         0.9  |       0.79 |         0.59 |  0.41 | mix_val_inter            |   4.98 |
| DeepJDOT    |         0.9  |       0.82 |         0.62 |  0.54 | prediction_entropy       |   2.92 |
| MCC         |         0.93 |       0.83 |         0.66 |  0.53 | mix_val_inter            |   2.38 |
| MDD         |         0.87 |       0.78 |         0.56 |  0.4  | mix_val_both             |   4.96 |
| SPA         |         0.91 |       0.78 |         0.56 |  0.41 | deep_embedded_validation |   5.39 |