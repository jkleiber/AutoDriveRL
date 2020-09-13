#!/bin/bash

# Pull the zip file.
wget -O ../DonkeySimLinux.zip https://github.com/tawnkramer/gym-donkeycar/releases/download/v20.8.31/DonkeySimLinux.zip

# Unzip the simulator.
unzip ../DonkeySimLinux.zip

# Delete the zip file.
rm -rf ../DonkeySimLinux.zip
