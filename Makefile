
SOURCE_GLOB=$(wildcard src/**/*.py tests/**/*.py models/**/.py)


.PHONY: lint
lint: pylint pycodestyle flake8 mypy

.PHONY: pylint
pylint:
	pylint \
		--load-plugins pylint_quotes \
		--disable=W0511,R0801,cyclic-import \
		$(SOURCE_GLOB)

.PHONY: pycodestyle
pycodestyle:
	pycodestyle \
		--statistics \
		--count \
		--ignore="${IGNORE_PEP}" \
		$(SOURCE_GLOB)

.PHONY: flake8
flake8:
	flake8 \
		--ignore="${IGNORE_PEP}" \
		$(SOURCE_GLOB)

.PHONY: mypy
mypy:
	MYPYPATH=stubs/ mypy \
		$(SOURCE_GLOB)

.PHONY: pytype
pytype:
	pytype \
		-V 3.8 \
		--disable=import-error,pyi-error \
		src/
	pytype \
		-V 3.8 \
		--disable=import-error \
		examples/

.PYHONY: test
test:
	make lint
	pytest src/ models/

.PYONY: install
install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

.PHONY: docs
docs:
	mkdocs serve
