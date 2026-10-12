.PHONY: sync sync-full lint lint-all format format-all format-fix format-fix-all typecheck test cleanliness ci help

# GitHub CI / project gate: scientific core only
UV_GROUPS_CI := --group dev
PY_CORE := src scripts tests

# Local workstation: notebooks + optional engines / cloud / LLM
UV_GROUPS_LOCAL := --group dev --group biophysics --group notebooks --group cloud --group llm
PY_ALL := src scripts notebooks tests

help:
	@echo "Targets:"
	@echo "  make ci              Project gate matching GitHub Actions (dev sync + core ruff/ty/pytest)"
	@echo "  make test            pytest tests/"
	@echo "  make lint            Ruff lint on core ($(PY_CORE))"
	@echo "  make format          Ruff format check on core"
	@echo "  make typecheck       uv check (ty) on core paths in pyproject"
	@echo "  make cleanliness     Local visual/repo hygiene: ruff on notebooks + core"
	@echo "  make sync            uv sync --group dev (CI parity)"
	@echo "  make sync-full       uv sync with biophysics/notebooks/cloud/llm"
	@echo "  make format-fix      Apply Ruff formatting on core"
	@echo "  make format-fix-all  Apply Ruff formatting including notebooks"

sync:
	uv sync $(UV_GROUPS_CI)

sync-full:
	uv sync $(UV_GROUPS_LOCAL)

lint:
	uv run ruff check $(PY_CORE)

lint-all:
	uv run ruff check $(PY_ALL)

format:
	uv run ruff format --check $(PY_CORE)

format-all:
	uv run ruff format --check $(PY_ALL)

format-fix:
	uv run ruff format $(PY_CORE)

format-fix-all:
	uv run ruff format $(PY_ALL)

typecheck:
	uv check

test:
	uv run pytest tests/ -v

# Notebooks + full tree formatting/lint for local eyes (not required on GitHub)
cleanliness: lint-all format-all

ci: sync lint format typecheck test
