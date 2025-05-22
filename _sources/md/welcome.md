---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# GeMMM

GeMMM (Generalized Mobile Movement Model) is a Python package that allows users to sample hourly journey numbers across England, Scotland and Wales. By combining probabilistic models with mobile telecoms data, GeMMM accounts for variability in these movement patterns and therefore reduces reliance on static datasets that only provide a snapshot at a single point in time. The {doc}`User Guide <../md/intro>` provides a more detailed description of the underlying models and data, along with caveats that users should be aware of before using this package, and the {doc}`Tutorial <../notebooks/example>` outlines the main functionality of GeMMM. The package source is freely available and hosted on [GitHub](https://github.com/ukhsa-collaboration/gemmm).

## Installation
The simplest way to install GeMMM is from PyPI using pip:
```
pip install gemmm
```

Alternatively, a copy of the source code can be downloaded from the GitHub repository:
```
git clone https://github.com/ukhsa-collaboration/gemmm.git
```
and the relevant dependencies, and GeMMM, can be installed:
```
cd gemmm
pip install -r requirements.txt
pip install .
```
The installation from source is successful if the following tests produce no errors:
```
python -m unittest discover -s tests
```

## Basic usage
Suppose that you want to simulate the number of journeys that are made between two areas at 8am on a weekday, this can be achieved with the following few lines of code:
```{code-cell} python3
:tags: [remove-input]

import pooch
pooch.get_logger().setLevel('WARNING')

import numpy
numpy.random.seed(1001)
```
```{code-cell} ipython3
import gemmm

start_msoa = 'E02000001'
end_msoa = 'E02000002'

sampler = gemmm.OriginDestination(msoas=[start_msoa, end_msoa], day_type='weekday')
sampler.generate_sample(hours=8)
sampler.to_pandas(hour=8, wide=True)
```

## Building the documentation locally
To build this documentation locally, first clone the GitHub repository and install the additional packages:
```
git clone https://github.com/ukhsa-collaboration/gemmm.git
cd gemmm
pip install -r docs/requirements.txt
```
Then, build the documentation from the command line:
```
jupyter-book build docs
```
The documentation can then be accessed from
```
docs/_build/html/index.html
```