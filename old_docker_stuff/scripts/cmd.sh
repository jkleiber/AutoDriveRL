#!/bin/bash

# Use the script's working directory
cd "$(dirname "${BASH_SOURCE[0]}")"

# Get configuration
source docker_config.sh

LAUNCH_ARGS="$@"

docker exec -t $CONTAINER conda init bash
docker exec -it $CONTAINER "${LAUNCH_ARGS}"
