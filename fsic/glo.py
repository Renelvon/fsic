"""A global module containing functions for managing the project."""

__author__ = "wittawat"

import os

try:
    import cPickle as pickle
except ImportError:
    import pickle

import fsic


def get_root():
    """Return the full path to the root of the package"""
    return os.path.abspath(os.path.dirname(fsic.__file__))


def result_folder():
    """Return the full path to the result/ folder containing experimental result
    files"""
    return os.path.join(get_root(), "result")


def ex_result_folder(ex):
    """Return the full path to the folder containing result files of the specified
    experiment.
    ex: a positive integer. """
    rp = result_folder()
    fpath = os.path.join(rp, "ex%d" % ex)
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    return fpath


def ex_result_file(ex, *relative_path):
    """Return the full path to the file identified by the relative path as a list
    of folders/files under the result folder of the experiment ex. """
    rf = ex_result_folder(ex)
    return os.path.join(rf, *relative_path)


def ex_load_result(ex, *relative_path):
    """Load a result identified by the  path from the experiment ex"""
    fpath = ex_result_file(ex, *relative_path)
    return pickle_load(fpath)


def pickle_load(fpath):
    if not os.path.isfile(fpath):
        raise ValueError("%s does not exist" % fpath)

    with open(fpath, "r") as f:
        # expect a dictionary
        result = pickle.load(f)
    return result
