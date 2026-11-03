"""Detach acyclic JSON report trees without generic object-graph traversal."""
from copy import deepcopy
import pickle


def copy_report(value):
    if type(value) is dict:return {k:copy_report(v) for k,v in value.items()}
    if type(value) is list:return [copy_report(v) for v in value]
    if type(value) is tuple:return tuple(copy_report(v) for v in value)
    if value is None or type(value) in (str,int,float,bool):return value
    return deepcopy(value)


def compiled_copy_report(value):
    """Detach internally generated report trees in C, preserving float bits.

    Only our own freshly serialized bytes are loaded here, never external data.
    The recursive copier remains available for isolated performance comparison.
    """
    return pickle.loads(pickle.dumps(value,protocol=pickle.HIGHEST_PROTOCOL))
