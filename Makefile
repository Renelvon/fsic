NAME=fsic
PIP=pip
PYTHON=python3
SETUP=setup.py
TESTS=tests

.PHONY: all build build_wheel check clean dist distclean install installcheck test uninstall

all: build

build:
	$(PYTHON) $(SETUP) bdist

build_wheel:
	$(PYTHON) $(SETUP) bdist_wheel

check:
	black $(SETUP) $(NAME) $(TESTS)
	check-manifest
	pylint setup.py $(NAME) $(TESTS)
	pyroma -n 10 .

clean:
	git clean -xfd

dist:
	$(PYTHON) $(SETUP) sdist

distclean: clean

install: build
	$(PYTHON) $(SETUP) install --user

installcheck: test

test:
	$(PYTHON) -m unittest discover -s $(TESTS)

uninstall:
	$(PIP) uninstall -y $(NAME)
