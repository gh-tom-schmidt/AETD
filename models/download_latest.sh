#!/bin/bash

mkdir -p models

urls=(
    "https://arcxyon.com/wp-content/uploads/2025/08/yolo-detect-m_best_epochs-100_size-460-960_05-08-2025.zip"
    "https://arcxyon.com/wp-content/uploads/2025/08/yolo-cls-s_best_epochs-30_size-32-32_06-08-2025.zip"
    "https://arcxyon.com/wp-content/uploads/2025/08/yolo-seg-m_full-road_best_epochs-300_size-460-960_07-08-2025.zip"
)

for url in "${urls[@]}"; do
    filename=$(basename "$url" .zip)
    zipfile="$filename.zip"

    # Download
    wget "$url"

    # Unzip
    unzip "$zipfile"

    # Move .pt file to models/
    mv "$filename/$filename.pt" "models/"

    # Cleanup
    rm -rf "$zipfile" "$filename"
done

echo "All models downloaded to models/"