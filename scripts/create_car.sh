#!/bin/bash

# Usage
# $> ./create_car.sh src/car username

# Create a car in the specified directory.
donkey createcar --path "$1"

# After running this, change the owner appropriately
sudo chown -R $2 $1
sudo chgrp -R $2 $1
