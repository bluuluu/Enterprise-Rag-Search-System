.PHONY: install run docker-up eval smoke build-cpp bench-cpp

install:
	python3 -m pip install -r requirements.txt

build-cpp:
	python3 setup.py build_ext --inplace

run:
	uvicorn app.main:app --reload

docker-up:
	docker compose up --build

eval:
	python3 scripts/evaluate_retrieval.py --username victor_viewer --top-k 5

smoke:
	bash scripts/smoke_test.sh

bench-cpp:
	python3 scripts/benchmark_vector_engine.py --num-vectors 200000 --dim 384 --top-k 10 --num-queries 20
