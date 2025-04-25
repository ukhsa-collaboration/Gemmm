# GeMMM

GeMMM () is a Python package that allows users to sample hourly journey numbers across England, Scotland and Wales. By combining probabilistic models with mobile telecoms data, GeMMM accounts for variability in these movement patterns and therefore reduces reliance on static datasets that only provide a single snapshot. The {doc}`User Guide <../md/intro>` provides a more detailed description of the underlying models and data, along with caveats that users should be aware of before using this package, and the {doc}`Tutorial <../notebooks/example>` outlines the main functionality of GeMMM. The package source is freely available and hosted on [GitHub](https://github.com/ukhsa-collaboration/Gemmm).

## Installation

## Basic usage
Suppose that you want to simulate the number of journeys that are made between two areas at 8am on a weekday, this can be achieved with the following few lines of code:
```python
import gemmm

start_msoa = 'E02000001'
end_msoa = 'E02000002'

sampler = gemmm.OriginDestination(msoas=[start_msoa, end_msoa], day_type='weekday')
sampler.generate_sample(hours=8)
```

## Building the documentation locally
To build this documentation locally, first install the additional packages:
```
pip install -r docs/requirements.txt
```
Then, build the documentation from the command line:
```
jupyter-book build docs
```
The documentation can then be accessed from
```
gemmm/docs/_build/html/index.html
```