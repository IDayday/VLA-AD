SHELL := /bin/bash
.DEFAULT_GOAL := all
.DELETE_ON_ERROR:

PYTHON ?= python3
SEED ?= 2027
BOOTSTRAP_SAMPLES ?= 10000
CONFIDENCE ?= 0.95

export PYTHONDONTWRITEBYTECODE := 1

.PHONY: help all pipeline audit prepare canonical validate stats tables figures pdf manifest checks

help:
	@echo "Supplementary reproducibility targets"
	@echo "  make pipeline  - canonicalize, validate, compute statistics, tables, figures"
	@echo "  make all       - audit, pipeline, compile PDF, and hash final artifacts"
	@echo "  make audit     - repository-relative file inventory and SHA-256 manifest"
	@echo "  make pdf       - compile supplementary.tex with latexmk/pdflatex"
	@echo "Variables: PYTHON=$(PYTHON), SEED=$(SEED), BOOTSTRAP_SAMPLES=$(BOOTSTRAP_SAMPLES), CONFIDENCE=$(CONFIDENCE)"

all: manifest checks

pipeline: tables figures

audit:
	$(PYTHON) scripts/build_inventory.py --seed $(SEED)

prepare:
	$(PYTHON) scripts/prepare_verified_sources.py --seed $(SEED)

canonical: prepare
	$(PYTHON) scripts/canonicalize_results.py --seed $(SEED)

validate: canonical
	$(PYTHON) scripts/validate_dataset.py --seed $(SEED)

stats: validate
	$(PYTHON) scripts/compute_statistics.py \
		--seed $(SEED) \
		--bootstrap-samples $(BOOTSTRAP_SAMPLES) \
		--confidence $(CONFIDENCE)

tables: stats
	$(PYTHON) scripts/generate_tables.py --seed $(SEED)
	$(PYTHON) scripts/generate_appendix_assets.py --seed $(SEED) --dpi 300

figures: tables

pdf: tables figures
	@test -f supplementary.tex || { echo "supplementary.tex is missing" >&2; exit 2; }
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error supplementary.tex; \
	elif command -v pdflatex >/dev/null 2>&1; then \
		pdflatex -interaction=nonstopmode -halt-on-error -file-line-error supplementary.tex; \
		if test -f supplementary.aux && grep -q '\\bibdata' supplementary.aux; then \
			command -v bibtex >/dev/null 2>&1 || { echo "bibtex is required but unavailable" >&2; exit 2; }; \
			bibtex supplementary; \
		fi; \
		pdflatex -interaction=nonstopmode -halt-on-error -file-line-error supplementary.tex; \
		pdflatex -interaction=nonstopmode -halt-on-error -file-line-error supplementary.tex; \
	else \
		echo "Neither latexmk nor pdflatex is installed" >&2; exit 2; \
	fi

manifest: audit pdf
	$(PYTHON) scripts/build_reproduction_manifest.py --seed $(SEED)

checks: manifest
	@for script in scripts/build_inventory.py scripts/canonicalize_results.py \
		scripts/validate_dataset.py scripts/compute_statistics.py \
		scripts/generate_tables.py scripts/generate_figures.py \
		scripts/prepare_verified_sources.py scripts/generate_appendix_assets.py \
		scripts/build_reproduction_manifest.py; do \
		$(PYTHON) "$$script" --help >/dev/null || exit 2; \
	done
	@if rg -n '/mnt/project|/mnt/navsim|internal server|TBD|TODO' --glob '*.tex' .; then \
		echo "Forbidden placeholder or internal path found in LaTeX" >&2; exit 2; \
	fi
	@if test -f supplementary.log && rg -n 'undefined references|Citation .* undefined|Reference .* undefined' supplementary.log; then \
		echo "Undefined citation/reference found in LaTeX log" >&2; exit 2; \
	fi
