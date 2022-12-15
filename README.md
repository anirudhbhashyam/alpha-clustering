# Alpha Clustering

[![Docs Build](https://readthedocs.org/projects/alpha-clustering/badge/?version=latest)](https://alpha-clustering.readthedocs.io/en/latest/?badge=latest)

![test](https://github.com/anirudhbhashyam/alpha-clustering/actions/workflows/test.yml/badge.svg)

This repository contains packages to run an algorithm to cluster data using alpha shapes.


# Documentation
The documentation can be found [here](https://alpha-clustering.readthedocs.io/en/latest/).


# Usage 

## Getting Started
To use the repository, the ideal thing would be to clone the repository locally and then install it as a package.
```
> python -m virtualenv venv
> source (.) venv/bin/activate
> mkdir deps
> git clone https://github.com/anirudhbhashyam/alpha-clustering deps
> cd deps/alpha-clustering
> pip install -e .
```

## Examples
```python
import numpy as np
from alpha_clustering.alpha_complex import AlphaComplexND
from alpha_clustering.cluster import Cluster

# Generate a random set of points in 2D.
points = np.random.rand(100, 2)
ac = AlphaComplexND(points)
# Fit the complex.
ac.fit()
# Predict the complex using some alpha.
ac.predict(alpha = 0.5)
# Setup the clustering object.
clustering = Cluster(ac.get_shape)
clustering.fit()
# Predict the clusters.
clusters = clustering.predict()
# The returned clusters are a list of lists of indices.
# So len(clusters) is the number of clusters.
```
For a more detailed guide visit the [documentation](https://alpha-clustering.readthedocs.io/en/latest/).


# Development
To run the tests, do the following from the root.
```
> pip install -r requirements.txt
> pytest
```

To build the documentation locally, do the following from the root.
```
> cd docs
> pip install -r requirements.txt
> make html
```
