.PHONY: install train serve docker-build docker-run

install:
	pip install -r requirements.txt

train:
	python3 -m src.ames.training.training_pipeline configs/pipeline.yaml

serve:
	uvicorn app.main:app --reload --port 8000

docker-build:
	docker build -t ames-housing-demo -f docker/Dockerfile .

docker-run:
	docker run --rm -p 8080:8080 ames-housing-demo
