"""
utils that denpend only on std
"""
from dataclasses import fields
from typing import Dict

import numpy as np


def intNone(s):
    if s == "":
        return None
    return int(s)


def str2slice(str_slice: str) -> slice:
    """
    convert a slice like string to a slice

    # examples:

    >>> assert str2slice(":2") == slice(start=None, stop=2)
    >>> assert str2slice("1:10:2") == slice(start=1, stop=10, step=2)

    """
    try:
        return int(str_slice)
    except ValueError:
        str_parts = (str_slice.split(":") + ["", "", ""])[:3]
        args = [intNone(s) for s in str_parts]

        return slice(*args)


def meanstd(x, axis=1, transpose=True):
    y = np.array([x.mean(axis), x.std(axis)])
    if transpose:
        return y.transpose()
    return y


def shallow_dict(obj) -> Dict:
    """
    shallow converstion of a dataclass to dict
    """
    return dict((field.name, getattr(obj, field.name)) for field in fields(obj))
