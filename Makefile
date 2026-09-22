# pip-tools flow: pyproject.toml is the source of truth for dependencies

.PHONY: lock sync install lint test check

lock:
	pip-compile --quiet --generate-hashes --strip-extras --output-file=requirements.txt pyproject.toml
	pip-compile --quiet --generate-hashes --strip-extras --extra=dev \
		--output-file=requirements-dev.txt pyproject.toml

sync:
	pip-sync requirements-dev.txt
	# pip-sync drops the editable install, so put it back without touching pinned deps
	pip install --no-deps -e .

install: sync

lint:
	ruff check src tests
	black --check src tests
	mypy src

test:
	pytest

check: lint test
