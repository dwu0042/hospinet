"""hospinet: Temporal Networks of Hospitals Using Patient Transfers

This package provides utilities for cleaning a database of patient admissions, especially to remove overlapping admissions, and for generating a temporal network of the aggregated movements of the implied transfers.

This takes heavy inspiration from the HospitalNetwork R package, and is intended to be a Python port of its checkBase functionality.
"""

__version__ = "1.0.2"


__all__ = [
    "temporal_network",
    "cleaner",
    "overlap_fixer",
    "TemporalNetwork",
]

from . import temporal_network, cleaner, overlap_fixer

from .temporal_network import TemporalNetwork


# private logging setup
import logging as __logging


def __create_logger():
    logger = __logging.getLogger("hospinet")
    logger.setLevel(__logging.WARNING)
    console_handler = __logging.StreamHandler()
    console_handler.setLevel(__logging.INFO)
    log_format = __logging.Formatter("%(levelname)s::%(name)s::%(message)s")
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    return logger


__logger = __create_logger()
