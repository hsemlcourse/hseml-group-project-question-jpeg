.PHONY: lint test install

install:
	pip install -r requirements.txt

lint:
	flake8 src/ tests/

test:
	pytest tests/

run-baseline:
	python src/train_baseline.py

run-experiments:
	python src/train_models.py
	python src/run_pca.py

all: install lint test run-baseline run-experiments