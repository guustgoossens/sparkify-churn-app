IMAGE ?= churn-explorer

.PHONY: install run test lint format docker-build docker-run sample

install:  ## Install the locked dependencies (incl. dev tools)
	uv sync --frozen

run:  ## Start the app on http://localhost:8501
	uv run streamlit run app.py

test:  ## Unit + app tests with coverage
	uv run pytest

lint:  ## Static checks (same as CI)
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8501:8501 $(IMAGE)

sample:  ## Rebuild data/sample_events.parquet, e.g. make sample SOURCE=~/train.parquet
	uv run python scripts/make_sample.py $(SOURCE)
