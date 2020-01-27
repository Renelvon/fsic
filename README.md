# The Finite Set Independence Criterion (FSIC)

[![Build Status](https://travis-ci.org/Renelvon/fsic.svg?branch=master)](https://travis-ci.org/Renelvon/fsic)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/Renelvon/fsic-test/blob/master/LICENSE)

This repository contains a Python 3.5 implementation of the normalized FSIC (NFSIC)
test as described in [our paper](https://arxiv.org/abs/1610.04782):

- Wittawat Jitkrittum, Zoltán Szabó and Arthur Gretton. **An Adaptive Test of Independence with Analytic Kernel Embeddings**. arXiv, 2016. 

## Installation

### Dependencies

`fsic` requires the following dependencies to run:

- [numpy](https://pypi.org/project/numpy/)
- [scipy](https://pypi.org/project/scipy/)
- [Theano](https://pypi.org/project/Theano/)

`numpy` and `scipy` will be automatically installed if you obtain `fsic` from PyPi.

Note that `theano` has some dependencies not contained within PyPi; see [this
page](http://deeplearning.net/software/theano/install.html#basic-user-install-instructions)
for how to install it properly.

### Installing `fsic` from PyPI

To install the package through [PyPI](https://pypi.org/), type:

```sh
$ pip3 install fsic --user
```

It is recommended that version `10` or later of `pip` is used; newer versions
of `pip` can handle missing dependencies better, by automatically downloading
and installing them together with `fsic`. To see which version of `pip` is
installed, type:

```sh
$ pip3 --version
```

If you are using an older version, you can locally upgrade `pip` by typing:

```
$ python3 -m pip install --user --upgrade-pip
```

before attempting the `fsic` installation.

### Build-Installing `fsic` from main repository
Alternatively, the package can be downloaded from source as follows:

1. Download the latest source code from GitHub (please *only* use the `master`
   branch when doing so, other branches are not guaranteed to contain a correct
   build):

```sh
$ git clone https://github.com/Renelvon/fsic.git
```

2. Install the package for this user only (showed by `--user`; installing the
   package system-wide might require `sudo -H`):

```sh
$ python3 setup.py install --user
```

### Confirming installation
Check that the package installed successfully by openning a new Python shell,
and issuing `import fsic`. If there is no import error, the installation
has completed successfully.

## Usage

The `tutorial` folder contains [Jupyter notebooks](https://jupyter.org/) that should help you start using the package. First, check [demo_nfsic.ipynb](https://github.com/Renelvon/fsic/blob/master/tutorial/demo_nfsic.ipynb); it will guide you through from the beginning. There are Jupyter notebooks in `tutorial` folder that you can explore to understand the software better.

## License
[MIT license](https://github.com/Renelvon/fsic-test/blob/master/LICENSE).

If you have questions or comments about anything regarding this work or code,
please do not hesitate the maintainer of this package ([Nikolaos Korasidis](nkorasid@student.ethz.ch)) or the original author ([Wittawat Jitkrittum](http://wittawat.com)).
