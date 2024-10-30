"""hospinet: Temporal Networks of Hospitals Using Patient Transfers

This package provides utilities for cleaning a database of patient admissions, especially to remove overlapping admissions, and for generating a temporal network of the aggregated movements of the implied transfers.

This takes heavy inspiration from the HospitalNetwork R package, and is intended to be a Python port of its checkBase functionality.
"""

__all__ = [
    "temporal_network",
    "cleaner",
    "overlap_fixer",
]

from . import *

from .temporal_network import TemporalNetwork
