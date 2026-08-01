PYTHON ?= python
ROOT := $(abspath .)
REPO ?= $(abspath ../..)
export PYTHONPATH := $(ROOT)/src:$(REPO):$(PYTHONPATH)

.PHONY: help test lint dry-run scope check package

help:
	@echo "make check    Run unit, syntax, dry-run, scope, and anonymity checks"
	@echo "make package  Create ../ampt_evaluation_release.zip"

test:
	$(PYTHON) -m pytest tests

lint:
	$(PYTHON) -m compileall -q src tests
	@for script in scripts/*.sh; do bash -n "$$script"; done

dry-run:
	@./scripts/evaluate_navsim_v1.sh >/dev/null
	@./scripts/export_single_trajectory_predictions.sh >/dev/null
	@./scripts/evaluate_navsim_v2.sh >/dev/null
	@./scripts/verify_reproduction.sh >/dev/null
	@for module in ampt_stage3.score_navsim_v2 \
		ampt_stage3.submission_to_manifest ampt_stage3.validate_checkpoint \
		ampt_stage3.verify_results ampt_stage3.build_release; do \
		$(PYTHON) -m $$module --help >/dev/null; \
	done

scope:
	$(PYTHON) tests/audit_release_scope.py

check: lint test dry-run scope

package: check
	./scripts/build_package.sh
