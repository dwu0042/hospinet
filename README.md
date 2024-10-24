# hospinet: Temporal Networks of Hospitals Using Patient Transfers

This package provides utilities for cleaning a database of patient admissions, especially to remove overlapping admissions, and for generating a temporal network of the aggregated movements of the implied transfers.

This takes heavy inspiration from the [HospitalNetwork](https://pascalcrepey.github.io/HospitalNetwork/) [R package](https://cran.r-project.org/package=HospitalNetwork), and is intended to be a Python port of its `checkBase` functionality.

## Installation

Install this using pip from pypi

```
pip install hospinet
```

or from source.