POETRY_RUN := poetry run
FOLDERS = tests src runners
export PROJECT_NAME=project_cookiecutter
export GITHUB_REPO=https://github.com/project-cookiecutter.git
export DATABRICKS_HOST=https://prod-datascience.cloud.databricks.com/


.PHONY: precommit poetry-precommit shell test \
	gitpod-init install install-dev report-coverage \
	lint lint-flake8 autolint lint-mypy lint-lite \
	lint-lite-flake8

gitpod-init: install

install: install-dev
	poetry install

install-dev:
	cp tools/pre-commit .git/hooks
	chmod +x .git/hooks/pre-commit

autolint:
	${POETRY_RUN} black ${FOLDERS}
	${POETRY_RUN} unify -r -i ${FOLDERS}
	${POETRY_RUN} isort ${FOLDERS}

hooks-off:
	python -m tools.cruft_editor .cruft.json run_hooks False

lint-lite:
	make autolint
	make lint-lite-flake8

lint:
	make autolint
	make lint-flake8
	make lint-mypy

lint-lite-flake8:
	@echo "\nRunning a flake8-lite...\n"
	${POETRY_RUN} flake8 --ignore=B,C,D,E,DAR,I,N,P,Q,RST,S,T,W .

lint-flake8:
	@echo "\nRunning flake8...\n"
	${POETRY_RUN} flake8 .

lint-mypy:
	@echo "\nRunning mypy...\n"
	${POETRY_RUN} mypy --show-error-codes src

precommit: poetry-precommit lint-lite

poetry-precommit:
	poetry run pre-commit run --all-files

report-coverage:
	${POETRY_RUN} coverage report
	${POETRY_RUN} coverage html
	${POETRY_RUN} coverage xml

shell:
	poetry shell

test:
	${POETRY_RUN} coverage erase
	${POETRY_RUN} coverage run --branch -m pytest tests \
		--junitxml=junit/test-results.xml \
		--doctest-modules -v
