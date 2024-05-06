## Phishing Website

This folder contains "phishing.txt", the data from the Phishing Website dataset originally provided in [1].

Originally, all the features are categorical. We recovered the preprocessed data from [2].

They can be downloaded from _<https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing>.

Binary encoding was used to generate feature vectors.

Each feature vector was normalized to maintain unit-length.

There are 4 domains: Books, DVDs, Electronics, and Kitchen appliances.

Each domain has around 2000 samples belongs to 2 classes: positive or negative.

For reference, please refer to:

[1] R. Mohammad, F. Thabtah, L. Mccluskey.
    An assessment of features related to phishing websites using an automated technique
    In International Conference for Internet Technology and Secured Transactions, 2012

[2] Yuchin Juan, Yong Zhuang, Wei-Sheng Chin, and Chih-Jen Lin.
    Field-aware factorization machines for CTR prediction.
    In Proceedings of the ACM Recommender Systems Conference (RecSys), 2016.
