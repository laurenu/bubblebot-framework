#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Install/update all packages from requirements-dev.txt
pip install -r requirements-dev.txt
