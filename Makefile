SHELL := /bin/bash
JULIA ?= julia
.DEFAULT_GOAL := help

# The variables CI sets: the items tagged `:slow` run, and FastCholesky throws on a matrix that is
# not symmetric instead of warning, so a local run fails where CI would.
CI_ENV := TEST_ALL=true JULIA_FASTCHOLESKY_THROW_ERROR_NON_SYMMETRIC=1

# Optional selection, passed to the suite: `make test test_args="name:rules"`.
test_args ?=
TEST_ARGS := $(if $(test_args),test_args = split("$(test_args)") .|> string,)

.PHONY: help test test-fast format check-format docs docs-serve clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

test: ## Run the suite as CI does; select with test_args="tag:<t> name:<n> <path>"
	$(CI_ENV) $(JULIA) --project=. -e 'import Pkg; Pkg.test($(TEST_ARGS))'

test-fast: ## Run the suite without the items tagged :slow
	$(JULIA) --project=. -e 'import Pkg; Pkg.test($(TEST_ARGS))'

format: ## Format the tree with Runic
	$(JULIA) -e 'import Pkg; Pkg.activate(temp = true); Pkg.add("Runic"); using Runic; exit(Runic.main(["--inplace", "."]))'

check-format: ## Check the formatting with Runic, without changing files
	$(JULIA) -e 'import Pkg; Pkg.activate(temp = true); Pkg.add("Runic"); using Runic; exit(Runic.main(["--check", "--diff", "."]))'

docs: ## Build the documentation into docs/build, running every example and doctest
	$(JULIA) --project=docs -e 'import Pkg; Pkg.instantiate()'
	$(JULIA) --project=docs docs/make.jl

docs-serve: ## Build and serve the documentation with live reload
	$(JULIA) --project=docs -e 'import Pkg; Pkg.instantiate(); using LiveServer; servedocs()'

clean: ## Remove the documentation build and coverage files
	rm -rf docs/build lcov.info
	find . \( -name '*.jl.cov' -o -name '*.jl.*.cov' -o -name '*.jl.mem' \) -delete
