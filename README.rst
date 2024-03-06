
Domain Adaptation Benchmark
=====================
|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to solver of domain adaptation problems.
In this problem, we consider source domain with labels and target domain without labels.
But shift happens:

.. math::
$$\mathcal{P}_s \neq \mathcal{P}_t,$$

with $\mathcal{P}_s$ and $\mathcal{P}_t$ the distributions of the source and target domains.
The goal is to learn a model that can predict the labels of the target domain using the source domain.

There exist different shifts:

.. math::
- Covariate shift: $\mathcal{P}_s(x) \neq \mathcal{P}_t(x)$ and $\mathcal{P}_s(y|x) = \mathcal{P}_t(y|x)$
- Label shift: $\mathcal{P}_s(y) \neq \mathcal{P}_t(y)$ and $\mathcal{P}_s(x|y) = \mathcal{P}_t(x|y)$
- Concept shift: $\mathcal{P}_s(x|y) = \mathcal{P}_t(x|y)$ and $\mathcal{P}_s(y|x) = \mathcal{P}_t(y|x)$
- Subspace shift: there exist a subspace $U$ such that $\mathcal{P}_s(UU^Tx) = \mathcal{P}_t(UU^Tx)$ and $\mathcal{P}_s(UU^Ty|x) = \mathcal{P}_t(UU^Ty|x)$

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/scikit-adaptation/skada-bench
   $ benchopt run skada-bench

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run skada-bench -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/scikit-adaptation/skada-bench/workflows/Tests/badge.svg
   :target: https://github.com/scikit-adaptation/skada-bench/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
