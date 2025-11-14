#!/bin/bash

set -e

# Configuration
IMAGE_NAME="td3-bc"
IMAGE_FULL="${IMAGE_NAME}:latest"
TAR_FILE="${IMAGE_NAME}.tar"
SIF_FILE="${IMAGE_NAME}.sif"

# Check for required commands
command -v docker >/dev/null 2>&1 || { echo >&2 "Docker is not installed or not in PATH."; exit 1; }
command -v apptainer >/dev/null 2>&1 || { echo >&2 "Apptainer is not installed or not in PATH."; exit 1; }

# Check if Docker image exists
if [[ "$(docker images -q $IMAGE_FULL 2> /dev/null)" == "" ]]; then
    echo "Docker image not found. Building from Dockerfile..."
    docker build -t $IMAGE_FULL .
else
    echo "Docker image '$IMAGE_FULL' already exists. Skipping build."
fi

# Check if tarball already exists
if [[ -f "$TAR_FILE" ]]; then
    echo "Tarball '$TAR_FILE' already exists."
    read -p "Do you want to overwrite it? (y/N): " choice
    case "$choice" in
        y|Y ) echo "Overwriting '$TAR_FILE'...";;
        * ) echo "Aborting."; exit 1;;
    esac
fi

# Save Docker image to tarball
echo "Saving Docker image to '$TAR_FILE', this may take a while..."
docker save $IMAGE_FULL -o $TAR_FILE

# Build Apptainer .sif image from Docker tarball
echo "Building Apptainer image: $SIF_FILE"
apptainer build $SIF_FILE docker-archive://$TAR_FILE

echo "Build complete."

read -p "Would you like to upload the image to the GPU cluster? (y/N): " choice
case "$choice" in
    y|Y )
        read -p "Enter your CIT username: " cluster_user
        if [[ -z "$cluster_user" ]]; then
            echo "No username entered. Aborting upload."
            exit 1
        fi

        echo "Uploading '$SIF_FILE' to /u/home/$cluster_user..."
        scp "$SIF_FILE" "${cluster_user}@vmknoll81.in.tum.de:/u/home/${cluster_user}/"
        echo "Upload complete."
        ;;
    * )
        echo "Upload skipped."
        exit 0
        ;;
esac
