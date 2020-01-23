NAME=fsic
PIP=pip
PYTHON=python2
SETUP=setup.py

.PHONY: all build check clean dist distclean install installcheck test uninstall

all: build

build:
	$(PYTHON) $(SETUP) build

check:
	check-manifest

clean:
	git clean -xfd

dist:
	$(PYTHON) $(SETUP) sdist

distclean: clean

install: build
	$(PYTHON) $(SETUP) install --user

installcheck: test

test:
	$(PYTHON) -m unittest discover -s fsic/test 

uninstall:
	$(PIP) uninstall -y $(NAME)
