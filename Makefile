.PHONY: sync lint format format-fix typecheck test ci help

UV_GROUPS := --group dev --group biophysics --group notebooks --group cloud --group llm
PY_PATHS := src scripts notebooks tests

help:
	@echo "Targets:"
	@echo "  make sync        Install/sync uv dependencies (dev + biophysics + notebooks + cloud + llm)"
	@echo "  make lint        Ruff lint on $(PY_PATHS)"
	@echo "  make format      Check Ruff formatting"
	@echo "  make format-fix  Apply Ruff formatting"
	@echo "  make typecheck   uv check (ty)"
	@echo "  make test        pytest tests/"
	@echo "  make ci          sync + lint + format + typecheck + test"

sync:
	uv sync $(UV_GROUPS)

lint:
	uv run ruff check $(PY_PATHS)

format:
	uv run ruff format --check $(PY_PATHS)

format-fix:
	uv run ruff format $(PY_PATHS)

typecheck:
	uv check

test:
	uv run pytest tests/ -v

ci: sync lint format typecheck test
