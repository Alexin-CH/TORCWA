# Define variables
VENV_DIR = venv
PYTHON = python3
PIP = $(VENV_DIR)/bin/pip

# Default target
all: install

# Create virtual environment
$(VENV_DIR):
	$(PYTHON) -m venv $(VENV_DIR)

# Install requirements and the torcwa package
install: $(VENV_DIR)
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

# Remove the virtual environment
clean:
	rm -rf $(VENV_DIR)

# Run the test suite (requires pytest in the active environment)
test:
	$(PYTHON) -m pytest tests/ -q

.PHONY: all install clean test
