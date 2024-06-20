.PHONY: install test train eval serve streamlit lint docker-build docker-up clean

PY ?= python

install:
	pip install -r requirements.txt

test:
	pytest -q

train:
	$(PY) -m src.train --config configs/default.yaml

eval:
	$(PY) -m src.eval --config configs/default.yaml --ckpt checkpoints/model.pt --gt-root data/raw/UCSD_Anomaly_Dataset/UCSDped2/Test

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

streamlit:
	streamlit run streamlit_app.py

lint:
	$(PY) -m pyflakes src tests || true

docker-build:
	docker build -t video-anomaly:dev .

docker-up:
	docker compose up --build

clean:
	rm -rf __pycache__ .pytest_cache build dist *.egg-info
	find . -name "*.pyc" -delete
