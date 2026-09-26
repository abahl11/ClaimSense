# Convenience targets for macOS/Linux or Windows with GNU make.
# On Windows without make, use the equivalent commands in run.ps1.
# Usage: make setup | make test | make lint | make format | make pipeline STAGE=validate | make app

ifeq ($(OS),Windows_NT)
    VENV_PY := .venv/Scripts/python.exe
else
    VENV_PY := .venv/bin/python
endif
BOOTSTRAP_PY ?= python
STAGE ?= all
CODE_DIRS := src tests app

.PHONY: setup test lint format pipeline app

setup:
	$(BOOTSTRAP_PY) -m venv .venv
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt

test:
	$(VENV_PY) -m pytest

lint:
	$(VENV_PY) -m ruff check $(CODE_DIRS)
	$(VENV_PY) -m black --check $(CODE_DIRS)

format:
	$(VENV_PY) -m ruff check --fix $(CODE_DIRS)
	$(VENV_PY) -m black $(CODE_DIRS)

pipeline:
	$(VENV_PY) -m src.pipeline --stage $(STAGE)

app:
	$(VENV_PY) -m streamlit run app/streamlit_app.py
