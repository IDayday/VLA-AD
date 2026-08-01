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
	$(PYTHON) scripts/build_inventory.py --seed $(SEED) \
		--scan-root docs --scan-root configs --scan-root scripts \
		--scan-root reports --scan-root navsim \
		--scan-root outputs/ampt_pdms914505_navtest_submission_20260730 \
		--scan-root outputs/pdms914487_numeric_anchor_reverse_old_residual_conflict_only_top20_graft0025_full_navtest_seed20260726_20260727/epoch_000_step_252/candidate/local_sharded_aggregated.csv \
		--scan-root outputs/fair_ec_pdms914153_navtest_20260727 \
		--scan-root outputs/navtest_hard_stable_zero658_recogdrive_stage3_vs_pdms914505_20260727 \
		--scan-root outputs/stage3_grpo_core_pareto_s16_lr1e4_b2acc4_20e_local8_20260616T020136Z

prepare:
	@if test -f ../outputs/ampt_pdms914505_navtest_submission_20260730/full_navtest_pdms.csv \
		&& test -f ../outputs/pdms914487_numeric_anchor_reverse_old_residual_conflict_only_top20_graft0025_full_navtest_seed20260726_20260727/epoch_000_step_252/candidate/local_sharded_aggregated.csv \
		&& test -f ../outputs/fair_ec_pdms914153_navtest_20260727/raw_path_temporal_cap2_final_score/navsim_v2_navtest_epdms.csv \
		&& test -f ../outputs/navtest_hard_stable_zero658_recogdrive_stage3_vs_pdms914505_20260727/per_token_comparison.csv; then \
		$(PYTHON) scripts/prepare_verified_sources.py --seed $(SEED); \
	elif test -f derived/source_adapters/v1_final.csv \
		&& test -f derived/source_adapters/v1_anchor.csv \
		&& test -f derived/source_adapters/v2_final.csv \
		&& test -f derived/source_adapters/hard658.csv; then \
		echo "Raw archive unavailable; using the packaged, hashed source adapters."; \
	else \
		echo "Neither the raw archive nor all packaged source adapters are available" >&2; exit 2; \
	fi

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
	@for script in launch/*.sh; do \
		bash -n "$$script" || exit 2; \
		"./$$script" --help >/dev/null || exit 2; \
		"./$$script" >/dev/null || exit 2; \
		done
	@bash -n make_supplement.sh
	@./make_supplement.sh --help >/dev/null
	@if launch/run_p0_pcmts.sh --execute >/dev/null 2>&1; then \
		echo "Training gate unexpectedly allowed --execute without ALLOW_TRAIN=1" >&2; exit 2; \
	fi
	@if rg -n '/mnt/project|/mnt/navsim|internal server|TBD|TODO' --glob '*.tex' .; then \
		echo "Forbidden placeholder or internal path found in LaTeX" >&2; exit 2; \
	fi
	@if test -f supplementary.log && rg -n 'undefined references|Citation .* undefined|Reference .* undefined' supplementary.log; then \
		echo "Undefined citation/reference found in LaTeX log" >&2; exit 2; \
	fi
	@if rg -n '\\bIPR\\b' supplementary.tex sections --glob '*.tex'; then \
		echo "Legacy IPR naming found in final LaTeX" >&2; exit 2; \
	fi
	@if test -f supplementary.log && rg -n 'Float too large|Overfull \\hbox \(([2-9][0-9]|[1-9][0-9]{2})' supplementary.log; then \
		echo "Material LaTeX overflow found" >&2; exit 2; \
	fi
