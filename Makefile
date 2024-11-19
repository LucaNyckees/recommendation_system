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

# Build the Docker images
docker-build:
	docker-compose build

# Start the services
docker-up:
	docker-compose up -d

# Stop the services
docker-down:
	docker-compose down

# Show logs for all services
docker-logs:
	docker-compose logs -f

# Run tests inside the app container (example)
docker-test:
	docker-compose run app pytest
