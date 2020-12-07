#!/bin/bash

# Use the src directory no matter what.
cd "$(dirname "${BASH_SOURCE[0]}")"

# Sync the virtual environment
pipenv sync

# Fetch the simulator
# NOTE: this is the simulator as of Sept. 13 2020.
# Replace this link with a newer one if it is not available.
wget https://github.com/tawnkramer/gym-donkeycar/releases/download/v20.9.9/DonkeySimLinux.zip

# Unzip to the simulator directory
unzip DonkeySimLinux.zip
mv DonkeySimLinux simulator
rm DonkeySimLinux.zip

# Install any packages that failed initially
pipenv sync
