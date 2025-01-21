# Makefile

#################################################################################
# GLOBALS                                                                       #
#################################################################################
SHELL:=/bin/bash
python=python3.10
BIN=venv/bin/

all: up

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

# Download amazon datasets
download-data:
	python . download datasets

# Load downloaded datasets to postgres database
load-data:
	python . load datasets

embed-data:
	python . load embeddings

# Run sentiment classifiers pipeline
sentiment:
	python . sentiment classifiers

# Build the Docker images
build:
	docker compose build

# Start the services
up: build
	docker compose up -d

# Stop the services
down:
	docker compose down

# Show logs for all services
logs:
	docker compose -f docker-compose.yml logs -f

# Run tests inside the app container (example)
test:
	docker compose run app pytest
