.PHONY: install test lint preprocess train optimize evaluate serve clean

# ---- Installation ----
install:
	pip install -r requirements.txt

install-dev: install
	pip install ruff mypy pytest-cov

# ---- Qualité du code ----
lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	ruff format src/ tests/

# ---- Tests ----
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# ---- Pipeline ML ----
preprocess:
	python -m src.data.preprocessing \
		--input-dir data/raw/ \
		--output-dir data/processed/ \
		--target-size 512 \
		--apply-clahe \
		--apply-ben-graham \
		--n-workers 8

train:
	python -m src.models.train \
		--config configs/train_config.yaml \
		--experiment-name retinai-v2

optimize:
	python -m src.models.optimize \
		--model-path models/training/best_model.keras \
		--output-path models/optimized/ \
		--pruning-rate 0.3 \
		--quantize dynamic

evaluate:
	python -m src.evaluation.evaluate \
		--model-path models/optimized/ \
		--data-dir data/processed/test/ \
		--tta \
		--generate-report

# ---- Déploiement ----
serve:
	docker compose up --build api

mlflow:
	docker compose up mlflow

serve-local:
	uvicorn src.api.main:app --reload --port 8000

# ---- Nettoyage ----
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .mypy_cache .ruff_cache
