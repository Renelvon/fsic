#!/usr/bin/env python2

"""
Setup script for fsic package.
"""

import setuptools

import fsic


def main():
    setuptools.setup(
        version=fsic.__version__,
    )
    # Rest of options are specified in `setup.cfg`


if __name__ == '__main__':
    main()
