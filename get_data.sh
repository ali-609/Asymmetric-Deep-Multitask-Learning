#!/bin/bash

# Function to create directory if it doesn't exist
create_directory() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
    fi
}

# Function to download file if it doesn't exist
download_file() {
    if [ ! -f "$2" ]; then
        echo "Downloading $1..."
        curl -o "$2" "$1"
    else
        echo "$2 already exists. Skipping download."
    fi
}

# Function to extract file if it exists
extract_file() {
    if [ -f "$1" ]; then
        echo "Extracting $1..."
        tar -xf "$1" -C "Datasets/"
    else
        echo "$1 does not exist. Skipping extraction."
    fi
}



# Create directory if it doesn't exist
create_directory "Datasets"

# Download file if it doesn't exist
download_file "https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar_semantic_bboxes.tar" "Datasets/camera_lidar_semantic_bboxes.tar"
download_file "https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar_semantic_bus.tar" "Datasets/camera_lidar_semantic_bus.tar"

# Extract file if it exists
extract_file "Datasets/camera_lidar_semantic_bboxes.tar" "Datasets/camera_lidar_semantic_bboxes/"
extract_file "Datasets/camera_lidar_semantic_bus.tar" "Datasets/bus/"
