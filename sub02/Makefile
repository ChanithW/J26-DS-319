.PHONY: setup train dev dev-backend dev-frontend test openapi

setup:
	pip install -e ".[dev]"
	cd frontend && npm install

train:
	python -m component2.cli -v train --data data/ --out model_bundle_v3.joblib

dev-backend:
	python -m uvicorn component2.api:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

# backend and frontend together (two processes)
dev:
	$(MAKE) -j2 dev-backend dev-frontend

test:
	python -m pytest
	cd frontend && npm test

openapi:
	python -m component2.api.dump_openapi
	cd frontend && npm run gen:types
