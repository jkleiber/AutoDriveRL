#!/bin/bash

# Source the global docker options and donkey car environment.
source scripts/docker_config.sh
source scripts/donkey_env.sh

# Create the image and tag name
IMAGE_TAG="${IMAGE}:latest"

if [ "$1" = "dev" ]; then
    # Set the build directory
    BUILD_DIR="${DOCKER_PATH}"

    # Build the image
    docker build $BUILD_DIR -t $IMAGE_TAG
fi

# Stop and delete old containers
docker stop ${CONTAINER}
docker rm ${CONTAINER}

# Run a container with the image we just built
docker run \
    -d \
    -v "$(pwd):/workspace" \
    --net=host \
    ${DONKEY_ENV} \
    --name ${CONTAINER} \
    "${IMAGE_TAG}"
