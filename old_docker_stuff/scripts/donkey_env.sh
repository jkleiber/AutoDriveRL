#!/bin/bash

# Donkey Car variables
DONKEY_GYM=True
DONKEY_SIM_PATH="/workspace/DonkeySimLinux/donkey_sim.x86_64"
DONKEY_GYM_ENV_NAME="donkey-generated-track-v0"

# Add environment vars to docker
DONKEY_ENV="-e DONKEY_GYM=$DONKEY_GYM"
DONKEY_ENV="$DONKEY_ENV -e DONKEY_SIM_PATH=$DONKEY_SIM_PATH"
DONKEY_ENV="$DONKEY_ENV -e DONKEY_GYM_ENV_NAME=$DONKEY_GYM_ENV_NAME"
