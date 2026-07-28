"""Central configuration: where the GUAM data lives.

On a new machine, set the GUAM_MAT environment variable to point at
Data_Set_1.mat, e.g.:

    export GUAM_MAT=/path/to/GUAM/Challenge_Problems/Data_Set_1.mat

If unset, we fall back to a path relative to the repository root
(../GUAM/Challenge_Problems/Data_Set_1.mat), which matches the layout
produced by `git clone` of the GUAM repo next to this code.
"""
from __future__ import annotations
import os

_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "GUAM", "Challenge_Problems", "Data_Set_1.mat",
)

GUAM_MAT = os.environ.get("GUAM_MAT", _DEFAULT)
