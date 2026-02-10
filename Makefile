.PHONY: install run docker-up eval smoke

install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --reload

docker-up:
	docker compose up --build

eval:
	python scripts/evaluate_retrieval.py --username victor_viewer --top-k 5

smoke:
	bash scripts/smoke_test.sh
