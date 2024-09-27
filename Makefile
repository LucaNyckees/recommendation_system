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

# Delete the virtual environment.
clean:
	rm -rf venv

# Install Python dependencies.
requirements: venv
	echo ">>> Installing requirements..."
	$(BIN)python -m pip install --upgrade pip wheel
	$(BIN)python -m pip install -r requirements.txt

# Silencing commands
.SILENT: venv requirements clean