# Makefile

#################################################################################
# GLOBALS                                                                       #
#################################################################################
SHELL:=/bin/bash
python=python3.10
BIN=venv/bin/

all: requirements

#################################################################################
# COMMANDS                                                                      #
#################################################################################

# Create a virtual environment.
venv:
	echo ">>> Create environment..."
	$(python) -m venv venv
	echo ">>> Environment successfully created!"

# Install Python dependencies.
requirements: venv
	echo ">>> Installing requirements..."
	$(BIN)python -m pip install --upgrade pip wheel
	$(BIN)python -m pip install -r requirements.txt

# Variables
IMAGE_NAME = recommendation_system:latest
CONTAINER_NAME = recommendation_system_container

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run a development container
run-dev:
	docker run --rm -it \
		-e USER=$USER \
		-v `pwd`:/app \
		-p 8000:8000 \
		-p 8501:8501 \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME) /bin/bash

# Run the FastAPI app inside a container
run-fastapi:
	docker run --rm -it \
		-p 8000:8000 \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME) \
		uvicorn src.fastapi_app.main:app --host 0.0.0.0 --port 8000

# Run the Dashboard inside a container
run-dashboard:
	docker run --rm -it \
		-p 8501:8501 \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME) \
		python src/dashboard/dashboard.py

# Run the initial dataset commands in the container
download-datasets:
	docker run --rm -it \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME) \
		python recommendation_system download datasets

load-datasets:
	docker run --rm -it \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME) \
		python recommendation_system load datasets

# Stop and clean up all running containers
clean:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true
