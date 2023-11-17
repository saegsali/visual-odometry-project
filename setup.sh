#!/bin/bash

# Define dataset URLs
PARKING_DATASET_URL="https://rpg.ifi.uzh.ch/docs/teaching/2023/parking.zip"
KITTI_DATASET_URL="https://rpg.ifi.uzh.ch/docs/teaching/2023/kitti05.zip"
MALAGA_DATASET_URL="https://rpg.ifi.uzh.ch/docs/teaching/2023/malaga-urban-dataset-extract-07.zip"

# Extract the filename from a URL
get_filename_from_url() {
    echo "${1##*/}"
}

# Define dataset names
PARKING_DATASET=$(get_filename_from_url "$PARKING_DATASET_URL")
KITTI_DATASET=$(get_filename_from_url "$KITTI_DATASET_URL")
MALAGA_DATASET=$(get_filename_from_url "$MALAGA_DATASET_URL")


# Download and unzip datasets function
download_and_unzip() {
  local url=$1
  local zip_file=$(get_filename_from_url "$url")
  local data_dir=$2

  # Check if the URL exists
  if curl --head --silent --fail "$url" > /dev/null; then
    echo "Downloading $zip_file..."
    curl -O "$url"
    echo "Unzipping $zip_file..."
    unzip -q -o "$zip_file" -d "$data_dir"
    # Remove the zip file
    rm -f "$zip_file"
  else
    echo "The URL $url does not exist. Skipping download and extraction for $zip_file."
  fi
}

# Download and unzip each dataset
echo "Checking and downloading datasets..."
download_and_unzip "$PARKING_DATASET_URL" "data"
download_and_unzip "$KITTI_DATASET_URL" "data"
download_and_unzip "$MALAGA_DATASET_URL" "data"

echo "Datasets downloaded and extracted successfully:)"
