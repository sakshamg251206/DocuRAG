# Common development tasks. Run `make help` to list them.

PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin

# Export variables from .env (if present) to every command, including the UI.
-include .env
export

.DEFAULT_GOAL := help

.PHONY: help install api ui test lint format typecheck check clean docker-up docker-down

help: ## Show this help
	@grep -hE '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create a virtualenv and install all dependencies
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	@if [ "$$(uname -s)" = "Linux" ]; then \
		$(BIN)/pip install torch --index-url https://download.pytorch.org/whl/cpu; \
	fi
	$(BIN)/pip install -r requirements-dev.txt

api: ## Run the API with auto-reload on http://localhost:8000
	$(BIN)/uvicorn backend.main:app --reload --reload-dir backend --port 8000

ui: ## Run the web UI on http://localhost:8501
	$(BIN)/streamlit run frontend/app.py

test: ## Run the test suite
	$(BIN)/pytest

lint: ## Check code style
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format: ## Auto-format and fix lint issues
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

typecheck: ## Run static type checks
	$(BIN)/mypy

check: lint typecheck test ## Run every check that CI runs

docker-up: ## Build and start the full stack (Ollama + API + UI) with Docker Compose
	docker compose up --build -d

docker-down: ## Stop the Docker Compose stack
	docker compose down

clean: ## Remove caches (keeps the virtualenv and indexed documents)
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -not -path "./$(VENV)/*" -exec rm -rf {} +
