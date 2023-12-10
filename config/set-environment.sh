#!/bin/bash

# Set up Python virtual environment
python3 -m venv myenv
source myenv/bin/activate

# Install requirements
pip install -r requirements.txt
